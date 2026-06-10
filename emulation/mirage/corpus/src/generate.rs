//! Input materialization: turn [`InputSpec`]s into `.npy` files on disk.
//!
//! Three sources are supported:
//!
//! * Upstream `Formula` — `coeff * U[0,1) + offset`, matching the
//!   `iree_corpus.py` behaviour.
//! * Upstream / native `File` — copy an existing `.npy`.
//! * Native CEL `formula` — a CEL expression that evaluates to a generator
//!   spec map (`{dist, shape, ...}`) which is then sampled deterministically.

use std::path::{Path, PathBuf};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;

use crate::cel;
use crate::error::{CorpusError, Result};
use crate::model::{Case, InputSpec};
use crate::tensor::{Dtype, Tensor};

/// A generator description produced by a CEL `formula` expression.
#[derive(Debug, Clone, Deserialize)]
struct GenSpec {
    /// Distribution / fill kind.
    dist: String,
    /// Tensor shape. Optional for `eye` (uses `n`).
    #[serde(default)]
    shape: Vec<usize>,
    /// Square size for `eye`.
    n: Option<usize>,
    /// Lower bound for `uniform`.
    low: Option<f64>,
    /// Upper bound for `uniform`.
    high: Option<f64>,
    /// Mean for `normal`.
    mean: Option<f64>,
    /// Std-dev for `normal`.
    std: Option<f64>,
    /// Constant for `splat`.
    value: Option<f64>,
    /// Start for `arange`.
    start: Option<f64>,
    /// Step for `arange`.
    step: Option<f64>,
    /// Optional dtype override (otherwise taken from the input spec).
    dtype: Option<String>,
}

/// A materialized input: its on-disk path plus the tensor itself (kept so
/// the runner can compute `expected_output` formulas without re-reading).
pub struct MaterializedInput {
    /// Path of the written / copied `.npy` file.
    pub path: PathBuf,
    /// The tensor values.
    pub tensor: Tensor,
}

/// Materialize every input of `case` into `run_dir`.
pub fn materialize_inputs(case: &Case, run_dir: &Path) -> Result<Vec<MaterializedInput>> {
    std::fs::create_dir_all(run_dir).map_err(|source| CorpusError::Io {
        path: run_dir.to_path_buf(),
        source,
    })?;
    let mut out = Vec::with_capacity(case.inputs.len());
    for (index, spec) in case.inputs.iter().enumerate() {
        out.push(materialize_one(case, spec, index, run_dir)?);
    }
    Ok(out)
}

fn materialize_one(
    case: &Case,
    spec: &InputSpec,
    index: usize,
    run_dir: &Path,
) -> Result<MaterializedInput> {
    // 1. Upstream `Formula`.
    if let Some(f) = &spec.formula_upstream {
        let dtype = Dtype::parse(&f.dtype)?;
        let mut rng = seeded_rng(case.seed, index);
        let n: usize = f.shape.iter().product();
        let data: Vec<f64> = (0..n)
            .map(|_| f.coeff * rng.gen_range(0.0..1.0) + f.offset)
            .collect();
        let tensor = Tensor::new(f.shape.clone(), data, dtype)?;
        let path = run_dir.join(format!("input_{index}.npy"));
        tensor.write_npy(&path)?;
        return Ok(MaterializedInput { path, tensor });
    }

    // 2. Upstream / native `File`.
    let file_path = spec
        .file_upstream
        .as_ref()
        .map(|f| f.path.clone())
        .or_else(|| spec.file.clone());
    if let Some(rel) = file_path {
        let source = case.dir().join(&rel);
        let tensor = Tensor::read_npy(&source)?;
        let dest = run_dir.join(
            Path::new(&rel)
                .file_name()
                .map(|s| s.to_owned())
                .unwrap_or_else(|| format!("input_{index}.npy").into()),
        );
        tensor.write_npy(&dest)?;
        return Ok(MaterializedInput { path: dest, tensor });
    }

    // 3. Native CEL `formula`.
    if let Some(expr) = &spec.formula {
        let label = format!("inputs[{index}].formula");
        let ctx = cel::context_with(&params_vars(case))?;
        let json = cel::eval_to_json(&label, expr, &ctx)?;
        let spec_gen: GenSpec = serde_json::from_value(json)
            .map_err(|e| CorpusError::cel(&label, format!("invalid generator spec: {e}")))?;
        let dtype = Dtype::parse(
            spec_gen
                .dtype
                .as_deref()
                .or(spec.dtype.as_deref())
                .unwrap_or("float32"),
        )?;
        let tensor = sample(&spec_gen, dtype, case.seed, index, &label)?;
        let path = run_dir.join(format!("input_{index}.npy"));
        tensor.write_npy(&path)?;
        return Ok(MaterializedInput { path, tensor });
    }

    Err(CorpusError::invalid(
        case.path.clone(),
        format!("inputs[{index}] has no Formula/File/formula/file"),
    ))
}

fn params_vars(case: &Case) -> Vec<(String, serde_json::Value)> {
    case.params
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Deterministic per-(case, input) RNG.
fn seeded_rng(seed: u64, index: usize) -> ChaCha8Rng {
    // Mix the input index in so distinct inputs get distinct streams while
    // staying fully reproducible from the case seed.
    let mixed = seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    ChaCha8Rng::seed_from_u64(mixed)
}

fn sample(spec: &GenSpec, dtype: Dtype, seed: u64, index: usize, label: &str) -> Result<Tensor> {
    let mut rng = seeded_rng(seed, index);
    let shape = if spec.dist == "eye" {
        let n = spec
            .n
            .or_else(|| spec.shape.first().copied())
            .ok_or_else(|| CorpusError::cel(label, "eye requires `n`"))?;
        vec![n, n]
    } else {
        spec.shape.clone()
    };
    let n: usize = shape.iter().product();
    let data: Vec<f64> = match spec.dist.as_str() {
        "uniform" => {
            let low = spec.low.unwrap_or(0.0);
            let high = spec.high.unwrap_or(1.0);
            (0..n).map(|_| rng.gen_range(low..high.max(low + f64::EPSILON))).collect()
        }
        "normal" => {
            let mean = spec.mean.unwrap_or(0.0);
            let std = spec.std.unwrap_or(1.0);
            (0..n).map(|_| mean + std * standard_normal(&mut rng)).collect()
        }
        "splat" | "const" => {
            let v = spec.value.unwrap_or(0.0);
            vec![v; n]
        }
        "zeros" => vec![0.0; n],
        "ones" => vec![1.0; n],
        "arange" => {
            let start = spec.start.unwrap_or(0.0);
            let step = spec.step.unwrap_or(1.0);
            (0..n).map(|i| start + step * i as f64).collect()
        }
        "eye" => {
            let side = shape[1];
            (0..n)
                .map(|i| if i / side == i % side { 1.0 } else { 0.0 })
                .collect()
        }
        other => {
            return Err(CorpusError::cel(
                label,
                format!("unsupported distribution {other:?}"),
            ));
        }
    };
    Tensor::new(shape, data, dtype)
}

/// Standard normal via Box–Muller, using the supplied uniform RNG.
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::UpstreamFormula;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn upstream_formula_is_deterministic() {
        let mut case = base_case();
        case.seed = 7;
        case.inputs = vec![InputSpec {
            formula_upstream: Some(UpstreamFormula {
                dtype: "float32".into(),
                shape: vec![2, 2],
                coeff: 2.0,
                offset: 1.0,
            }),
            ..Default::default()
        }];
        let d1 = tmp();
        let d2 = tmp();
        let a = materialize_inputs(&case, d1.path()).unwrap();
        let b = materialize_inputs(&case, d2.path()).unwrap();
        assert_eq!(a[0].tensor.data, b[0].tensor.data);
        assert_eq!(a[0].tensor.shape, vec![2, 2]);
        // coeff*U+offset is within [offset, offset+coeff).
        for &v in &a[0].tensor.data {
            assert!((1.0..3.0).contains(&v));
        }
    }

    #[test]
    fn cel_uniform_formula() {
        let mut case = base_case();
        case.params
            .insert("n".into(), serde_json::json!(3));
        case.inputs = vec![InputSpec {
            dtype: Some("float32".into()),
            formula: Some("{'dist': 'uniform', 'shape': [n, n], 'low': 0.0, 'high': 1.0}".into()),
            ..Default::default()
        }];
        let d = tmp();
        let out = materialize_inputs(&case, d.path()).unwrap();
        assert_eq!(out[0].tensor.shape, vec![3, 3]);
        assert!(out[0].path.exists());
    }

    #[test]
    fn cel_eye_formula() {
        let mut case = base_case();
        case.inputs = vec![InputSpec {
            dtype: Some("float32".into()),
            formula: Some("{'dist': 'eye', 'n': 3}".into()),
            ..Default::default()
        }];
        let d = tmp();
        let out = materialize_inputs(&case, d.path()).unwrap();
        let t = &out[0].tensor;
        assert_eq!(t.at(&[0, 0]), Some(1.0));
        assert_eq!(t.at(&[1, 0]), Some(0.0));
        assert_eq!(t.at(&[2, 2]), Some(1.0));
    }

    fn base_case() -> Case {
        Case {
            name: "t".into(),
            kind: "run_module".into(),
            sources: vec!["a.mlir".into()],
            function: "main".into(),
            vmfb_names: vec![],
            compile_only: false,
            seed: 0,
            inputs: vec![],
            run_flags: vec![],
            outputs: vec![],
            expected_output: vec![],
            checks: vec![],
            validate: None,
            rtol: None,
            atol: None,
            compile_flags: vec![],
            params: Default::default(),
            path: PathBuf::from("/corpus/t/case.json"),
        }
    }
}
