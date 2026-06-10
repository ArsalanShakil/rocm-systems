//! Data model for corpus *cases*, *target configs*, and *scenarios*.
//!
//! These types deserialize from **both** the upstream JSON layout and the
//! native TOML layout. Fields that only exist in one format are optional
//! with sensible defaults, so a single set of structs covers both.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

fn default_kind() -> String {
    "run_module".to_string()
}

/// A value that may be written as a single item or a list of items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    /// A single value.
    One(T),
    /// A list of values.
    Many(Vec<T>),
}

impl<T: Clone> OneOrMany<T> {
    /// Flatten into a plain vector.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(v) => vec![v],
            OneOrMany::Many(v) => v,
        }
    }
}

/// The upstream `{"Formula": {...}}` input spec body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpstreamFormula {
    /// numpy-style dtype string, e.g. `"float32"`.
    pub dtype: String,
    /// Tensor shape.
    #[serde(default)]
    pub shape: Vec<usize>,
    /// Multiplier applied to a uniform `[0, 1)` sample.
    #[serde(default = "one_f64")]
    pub coeff: f64,
    /// Constant added after scaling.
    #[serde(default)]
    pub offset: f64,
}

fn one_f64() -> f64 {
    1.0
}

/// The upstream `{"File": {...}}` input spec body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpstreamFile {
    /// Path (relative to the case directory) of a `.npy` file.
    pub path: String,
}

/// Specification for a single kernel input.
///
/// Supports the upstream `Formula`/`File` shapes and the native
/// `dtype` + `formula` (CEL) / `file` shapes simultaneously.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InputSpec {
    /// Upstream random-tensor formula (`{"Formula": {...}}`).
    #[serde(rename = "Formula", default, skip_serializing_if = "Option::is_none")]
    pub formula_upstream: Option<UpstreamFormula>,

    /// Upstream file reference (`{"File": {...}}`).
    #[serde(rename = "File", default, skip_serializing_if = "Option::is_none")]
    pub file_upstream: Option<UpstreamFile>,

    /// Native dtype string for a CEL `formula` input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Native CEL expression that evaluates to a generator spec map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formula: Option<String>,

    /// Native file reference (path relative to the case directory).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

/// Specification for an expected output, used to derive the reference
/// tensor a [`Check`] or CEL `validate` predicate compares against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputSpec {
    /// Load the reference tensor from a `.npy` file in the case directory.
    File {
        /// Path relative to the case directory.
        path: String,
    },
    /// Compute the reference tensor from a closed-form op over inputs.
    Formula {
        /// The op to apply (currently `"matmul"`).
        op: String,
        /// Indices into the case `inputs` used as operands.
        inputs: Vec<usize>,
    },
}

/// A built-in numeric check (upstream `checks[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Check {
    /// Element-wise `np.allclose`.
    Allclose {
        /// Output index this check applies to.
        #[serde(default)]
        output: usize,
        /// Relative tolerance override.
        rtol: Option<f64>,
        /// Absolute tolerance override.
        atol: Option<f64>,
    },
    /// `allclose` restricted to a sparse list of coordinates.
    SparseAllclose {
        /// Output index this check applies to.
        #[serde(default)]
        output: usize,
        /// Coordinates to compare.
        indices: Vec<Vec<usize>>,
        /// Relative tolerance override.
        rtol: Option<f64>,
        /// Absolute tolerance override.
        atol: Option<f64>,
    },
    /// `allclose` restricted to a rectangular corner of the tensor.
    CornerAllclose {
        /// Output index this check applies to.
        #[serde(default)]
        output: usize,
        /// One of `top_left`, `top_right`, `bottom_left`, `bottom_right`.
        corner: String,
        /// Size of the corner window.
        shape: Vec<usize>,
        /// Relative tolerance override.
        rtol: Option<f64>,
        /// Absolute tolerance override.
        atol: Option<f64>,
    },
}

impl Check {
    /// The output index this check targets.
    pub fn output(&self) -> usize {
        match self {
            Check::Allclose { output, .. }
            | Check::SparseAllclose { output, .. }
            | Check::CornerAllclose { output, .. } => *output,
        }
    }
}

/// A kernel test case: a program to compile, inputs to feed it, and
/// expectations to validate its outputs against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Case {
    /// Unique case name.
    pub name: String,

    /// Case kind. Only `"run_module"` is currently supported.
    #[serde(default = "default_kind")]
    pub kind: String,

    /// MLIR source files (relative to the case directory).
    pub sources: Vec<String>,

    /// Entry-point function to invoke.
    pub function: String,

    /// One compiled module name per source. Derived from `sources` if empty.
    #[serde(default)]
    pub vmfb_names: Vec<String>,

    /// Compile the sources but never run them.
    #[serde(default)]
    pub compile_only: bool,

    /// Seed for deterministic input generation.
    #[serde(default)]
    pub seed: u64,

    /// Input specifications, materialized to `.npy` before running.
    #[serde(default)]
    pub inputs: Vec<InputSpec>,

    /// Extra `iree-run-module` flags.
    #[serde(default)]
    pub run_flags: Vec<String>,

    /// Output file names produced by the run.
    #[serde(default)]
    pub outputs: Vec<String>,

    /// Reference outputs used by [`Check`]s / CEL predicates.
    #[serde(default)]
    pub expected_output: Vec<OutputSpec>,

    /// Built-in numeric checks (upstream-compatible).
    #[serde(default)]
    pub checks: Vec<Check>,

    /// Native CEL validation predicates (each must evaluate to `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validate: Option<OneOrMany<String>>,

    /// Default relative tolerance.
    pub rtol: Option<f64>,

    /// Default absolute tolerance.
    pub atol: Option<f64>,

    /// Extra `iree-compile` flags.
    #[serde(default)]
    pub compile_flags: Vec<String>,

    /// Named parameters exposed to CEL `formula` / `validate` expressions.
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,

    /// Filesystem path the case was loaded from (filled by the loader).
    #[serde(skip)]
    pub path: PathBuf,
}

impl Case {
    /// Default relative tolerance (matching numpy's `allclose`).
    pub fn rtol(&self) -> f64 {
        self.rtol.unwrap_or(1e-5)
    }

    /// Default absolute tolerance (matching numpy's `allclose`).
    pub fn atol(&self) -> f64 {
        self.atol.unwrap_or(1e-8)
    }

    /// The directory the case lives in.
    pub fn dir(&self) -> PathBuf {
        self.path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default()
    }

    /// Native CEL validation predicates, flattened.
    pub fn validate_exprs(&self) -> Vec<String> {
        self.validate
            .clone()
            .map(OneOrMany::into_vec)
            .unwrap_or_default()
    }
}

/// A *target config*: the compile/run flags and skip/xfail lists that
/// define how a family of cases is built and executed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetConfig {
    /// Human-readable config name (used in result ids).
    pub config_name: String,

    /// Flags passed to `iree-compile` for every case.
    #[serde(default)]
    pub iree_compile_flags: Vec<String>,

    /// Flags passed to `iree-run-module` for every case.
    #[serde(default)]
    pub iree_run_module_flags: Vec<String>,

    /// Default run wrapper (overridden by the scenario / `--run-wrapper`).
    #[serde(default)]
    pub iree_run_module_wrapper: Vec<String>,

    /// Cases whose compile step is skipped entirely.
    #[serde(default)]
    pub skip_compile_tests: Vec<String>,

    /// Cases expected to fail compilation.
    #[serde(default)]
    pub expected_compile_failures: Vec<String>,

    /// Cases compiled but not run.
    #[serde(default)]
    pub skip_run_tests: Vec<String>,

    /// Cases expected to fail at run/validation time.
    #[serde(default)]
    pub expected_run_failures: Vec<String>,

    /// Filesystem path the config was loaded from (filled by the loader).
    #[serde(skip)]
    pub path: PathBuf,
}

/// A *scenario*: a named way to run a kernel — a mirage profile plus the
/// emulator backend it pins. Scenarios are what the UI/CLI iterate over
/// to compare "the same kernel under different conditions".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    /// Stable scenario name (e.g. `rocjitsu`, `hotswap`, `native`).
    pub name: String,

    /// Human-readable description.
    #[serde(default)]
    pub description: String,

    /// The emulator backend this scenario drives.
    pub emulator: String,

    /// The builtin GPU agent the profile pins.
    pub agent: String,

    /// The mirage profile name to create / reuse.
    pub profile: String,
}
