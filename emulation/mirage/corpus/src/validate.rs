//! Output validation.
//!
//! Two complementary mechanisms:
//!
//! * **Built-in checks** ([`Check`]) — `allclose`, `sparse_allclose`,
//!   `corner_allclose` — evaluated natively, matching `iree_corpus.py`.
//! * **CEL predicates** (the case `validate` list) — boolean expressions
//!   over precomputed *facts* about each observed/expected tensor pair,
//!   enabling property checks (means, ranges, finiteness, shapes, ...).

use crate::cel;
use crate::error::{CorpusError, Result};
use crate::model::{Case, Check, OutputSpec};
use crate::tensor::Tensor;

/// Compute the expected reference tensors from a case's `expected_output`.
pub fn expected_tensors(case: &Case, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    let mut out = Vec::with_capacity(case.expected_output.len());
    for spec in &case.expected_output {
        out.push(match spec {
            OutputSpec::File { path } => Tensor::read_npy(&case.dir().join(path))?,
            OutputSpec::Formula { op, inputs: idx } => match op.as_str() {
                "matmul" => {
                    if idx.len() != 2 {
                        return Err(CorpusError::Validation(
                            "matmul expected_output requires two inputs".into(),
                        ));
                    }
                    let a = inputs.get(idx[0]).ok_or_else(|| {
                        CorpusError::Validation(format!("matmul input {} missing", idx[0]))
                    })?;
                    let b = inputs.get(idx[1]).ok_or_else(|| {
                        CorpusError::Validation(format!("matmul input {} missing", idx[1]))
                    })?;
                    matmul(a, b)?
                }
                other => {
                    return Err(CorpusError::Validation(format!(
                        "unsupported expected_output op {other:?}"
                    )));
                }
            },
        });
    }
    Ok(out)
}

/// Validate observed outputs against expectations.
///
/// Returns `Ok(())` on success or [`CorpusError::Validation`] on the first
/// failing check/predicate.
pub fn validate(case: &Case, observed: &[Tensor], expected: &[Tensor]) -> Result<()> {
    let checks = effective_checks(case, observed.len());

    for check in &checks {
        let oi = check.output();
        let obs = observed
            .get(oi)
            .ok_or_else(|| CorpusError::Validation(format!("check references missing output {oi}")))?;
        let exp = expected.get(oi).ok_or_else(|| {
            CorpusError::Validation(format!("no expected output for index {oi}"))
        })?;
        run_check(case, check, obs, exp)?;
    }

    let predicates = case.validate_exprs();
    if !predicates.is_empty() {
        let vars = fact_vars(case, observed, expected);
        let ctx = cel::context_with(&vars)?;
        for (i, expr) in predicates.iter().enumerate() {
            let label = format!("validate[{i}]");
            if !cel::eval_bool(&label, expr, &ctx)? {
                return Err(CorpusError::Validation(format!(
                    "predicate failed: {expr}"
                )));
            }
        }
    }

    Ok(())
}

/// If a case lists no explicit checks/predicates but does provide expected
/// outputs, default to an `allclose` per output (matching upstream).
fn effective_checks(case: &Case, num_outputs: usize) -> Vec<Check> {
    if !case.checks.is_empty() {
        return case.checks.clone();
    }
    if case.validate_exprs().is_empty() && !case.expected_output.is_empty() {
        return (0..num_outputs)
            .map(|output| Check::Allclose {
                output,
                rtol: None,
                atol: None,
            })
            .collect();
    }
    Vec::new()
}

fn run_check(case: &Case, check: &Check, obs: &Tensor, exp: &Tensor) -> Result<()> {
    match check {
        Check::Allclose { rtol, atol, .. } => {
            let (rtol, atol) = tol(case, *rtol, *atol);
            if !allclose(&obs.data, &exp.data, rtol, atol) {
                return Err(CorpusError::Validation(format!(
                    "allclose failed (rtol={rtol}, atol={atol}, max_abs_diff={})",
                    max_abs_diff(&obs.data, &exp.data)
                )));
            }
        }
        Check::SparseAllclose {
            indices, rtol, atol, ..
        } => {
            let (rtol, atol) = tol(case, *rtol, *atol);
            for idx in indices {
                let o = obs.at(idx).ok_or_else(|| {
                    CorpusError::Validation(format!("sparse index {idx:?} out of bounds"))
                })?;
                let e = exp.at(idx).ok_or_else(|| {
                    CorpusError::Validation(format!("sparse index {idx:?} out of bounds"))
                })?;
                if !close(o, e, rtol, atol) {
                    return Err(CorpusError::Validation(format!(
                        "sparse_allclose failed at {idx:?}: observed={o}, expected={e}"
                    )));
                }
            }
        }
        Check::CornerAllclose {
            corner,
            shape,
            rtol,
            atol,
            ..
        } => {
            let (rtol, atol) = tol(case, *rtol, *atol);
            let slices = corner_slices(&obs.shape, corner, shape)?;
            for idx in iterate_box(&slices) {
                let o = obs.at(&idx).unwrap_or(f64::NAN);
                let e = exp.at(&idx).unwrap_or(f64::NAN);
                if !close(o, e, rtol, atol) {
                    return Err(CorpusError::Validation(format!(
                        "corner_allclose ({corner}) failed at {idx:?}: observed={o}, expected={e}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn tol(case: &Case, rtol: Option<f64>, atol: Option<f64>) -> (f64, f64) {
    (rtol.unwrap_or_else(|| case.rtol()), atol.unwrap_or_else(|| case.atol()))
}

fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn allclose(a: &[f64], b: &[f64], rtol: f64, atol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| close(x, y, rtol, atol))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
        return Err(CorpusError::Validation(format!(
            "matmul shape mismatch: {:?} x {:?}",
            a.shape, b.shape
        )));
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            data[i * n + j] = acc;
        }
    }
    Tensor::new(vec![m, n], data, a.dtype)
}

/// Compute `(start, len)` per axis for a corner window (mirrors upstream).
fn corner_slices(out_shape: &[usize], corner: &str, win: &[usize]) -> Result<Vec<(usize, usize)>> {
    if win.len() > out_shape.len() {
        return Err(CorpusError::Validation(format!(
            "corner rank {} exceeds output rank {}",
            win.len(),
            out_shape.len()
        )));
    }
    let mut slices = Vec::with_capacity(win.len());
    for (axis, &size) in win.iter().enumerate() {
        let dim = out_shape[axis];
        if size > dim {
            return Err(CorpusError::Validation(format!(
                "corner size {size} exceeds dimension {dim}"
            )));
        }
        let last = axis == win.len() - 1;
        let start = if (corner == "top_left" || corner == "top_right") || !last {
            if (corner == "top_right" || corner == "bottom_right") && last {
                dim - size
            } else {
                0
            }
        } else {
            dim - size
        };
        // Right corners pull the last axis to the right edge.
        let start = if (corner == "top_right" || corner == "bottom_right") && last {
            dim - size
        } else {
            start
        };
        slices.push((start, size));
    }
    Ok(slices)
}

/// Enumerate every multi-index inside a box defined by `(start, len)` axes.
fn iterate_box(slices: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for &(start, len) in slices {
        let mut next = Vec::new();
        for prefix in &result {
            for offset in 0..len {
                let mut idx = prefix.clone();
                idx.push(start + offset);
                next.push(idx);
            }
        }
        result = next;
    }
    result
}

/// Build the CEL variable set (facts) for the `validate` predicates.
fn fact_vars(case: &Case, observed: &[Tensor], expected: &[Tensor]) -> Vec<(String, serde_json::Value)> {
    let mut vars: Vec<(String, serde_json::Value)> = case
        .params
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let per_output: Vec<serde_json::Value> = observed
        .iter()
        .enumerate()
        .map(|(i, obs)| facts_map(case, obs, expected.get(i)))
        .collect();

    // Output 0's facts are also exposed at the top level for convenience.
    if let Some(first) = per_output.first()
        && let Some(obj) = first.as_object()
    {
        for (k, v) in obj {
            vars.push((k.clone(), v.clone()));
        }
    }
    vars.push(("outputs".into(), serde_json::Value::Array(per_output)));
    vars
}

fn facts_map(case: &Case, obs: &Tensor, exp: Option<&Tensor>) -> serde_json::Value {
    let n = obs.data.len().max(1) as f64;
    let sum: f64 = obs.data.iter().sum();
    let mean = sum / n;
    let min = obs.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = obs.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let all_finite = obs.data.iter().all(|v| v.is_finite());

    let mut map = serde_json::Map::new();
    map.insert("count".into(), serde_json::json!(obs.data.len()));
    map.insert("shape".into(), serde_json::json!(obs.shape));
    map.insert("sum".into(), json_f64(sum));
    map.insert("mean".into(), json_f64(mean));
    map.insert("min".into(), json_f64(if obs.is_empty() { 0.0 } else { min }));
    map.insert("max".into(), json_f64(if obs.is_empty() { 0.0 } else { max }));
    map.insert("all_finite".into(), serde_json::json!(all_finite));

    if let Some(exp) = exp {
        let (rtol, atol) = (case.rtol(), case.atol());
        let mad = max_abs_diff(&obs.data, &exp.data);
        let mrd = obs
            .data
            .iter()
            .zip(&exp.data)
            .map(|(&x, &y)| if y == 0.0 { (x - y).abs() } else { (x - y).abs() / y.abs() })
            .fold(0.0, f64::max);
        let mismatches = obs
            .data
            .iter()
            .zip(&exp.data)
            .filter(|&(&x, &y)| !close(x, y, rtol, atol))
            .count();
        map.insert("max_abs_diff".into(), json_f64(mad));
        map.insert("max_rel_diff".into(), json_f64(mrd));
        map.insert("mismatches".into(), serde_json::json!(mismatches));
        map.insert(
            "allclose".into(),
            serde_json::json!(allclose(&obs.data, &exp.data, rtol, atol)),
        );
    }
    serde_json::Value::Object(map)
}

fn json_f64(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OneOrMany;
    use crate::tensor::Dtype;
    use std::path::PathBuf;

    fn case_with(checks: Vec<Check>, validate: Option<Vec<String>>) -> Case {
        Case {
            name: "t".into(),
            kind: "run_module".into(),
            sources: vec![],
            function: "main".into(),
            vmfb_names: vec![],
            compile_only: false,
            seed: 0,
            inputs: vec![],
            run_flags: vec![],
            outputs: vec![],
            expected_output: vec![],
            checks,
            validate: validate.map(OneOrMany::Many),
            rtol: Some(1e-3),
            atol: Some(1e-5),
            compile_flags: vec![],
            params: Default::default(),
            path: PathBuf::from("/c/case.json"),
        }
    }

    fn t(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
        Tensor::new(shape, data, Dtype::F32).unwrap()
    }

    #[test]
    fn allclose_pass_and_fail() {
        let case = case_with(vec![Check::Allclose { output: 0, rtol: None, atol: None }], None);
        let obs = t(vec![2], vec![1.0, 2.0]);
        let exp = t(vec![2], vec![1.0, 2.0005]);
        assert!(validate(&case, &[obs], &[exp]).is_ok());

        let bad = t(vec![2], vec![1.0, 5.0]);
        let exp2 = t(vec![2], vec![1.0, 2.0]);
        assert!(validate(&case, &[bad], &[exp2]).is_err());
    }

    #[test]
    fn sparse_check() {
        let case = case_with(
            vec![Check::SparseAllclose {
                output: 0,
                indices: vec![vec![0, 0], vec![1, 1]],
                rtol: None,
                atol: None,
            }],
            None,
        );
        let obs = t(vec![2, 2], vec![1.0, 9.0, 9.0, 4.0]);
        let exp = t(vec![2, 2], vec![1.0, 0.0, 0.0, 4.0]);
        // Only diagonal is checked, off-diagonal differences are ignored.
        assert!(validate(&case, &[obs], &[exp]).is_ok());
    }

    #[test]
    fn corner_top_left() {
        let case = case_with(
            vec![Check::CornerAllclose {
                output: 0,
                corner: "top_left".into(),
                shape: vec![1, 1],
                rtol: None,
                atol: None,
            }],
            None,
        );
        let obs = t(vec![2, 2], vec![5.0, 9.0, 9.0, 9.0]);
        let exp = t(vec![2, 2], vec![5.0, 0.0, 0.0, 0.0]);
        assert!(validate(&case, &[obs], &[exp]).is_ok());
    }

    #[test]
    fn cel_property_predicate() {
        let case = case_with(
            vec![],
            Some(vec!["all_finite && mean > 0.0 && shape == [2]".into()]),
        );
        let obs = t(vec![2], vec![1.0, 3.0]);
        assert!(validate(&case, &[obs], &[]).is_ok());

        let case2 = case_with(vec![], Some(vec!["max_abs_diff < 0.01".into()]));
        let obs = t(vec![2], vec![1.0, 2.0]);
        let exp = t(vec![2], vec![1.0, 2.0]);
        assert!(validate(&case2, &[obs], &[exp]).is_ok());
    }

    #[test]
    fn matmul_expected() {
        let a = t(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = t(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
