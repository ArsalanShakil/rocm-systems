//! mirage_corpus — a native runner for the rocjitsu test corpus.
//!
//! This crate loads kernel test *cases* (compiled-then-run MLIR programs),
//! generates their inputs, runs them through a mirage emulator *scenario*
//! (rocjitsu CPU functional model, HotSwap on a real GPU, or native
//! pass-through), and validates the observed outputs.
//!
//! It deliberately understands two on-disk formats:
//!
//! * The upstream [`ROCm/rocjitsu-test-corpus`] JSON layout (the IREE test
//!   suite convention: one JSON per *case* plus one JSON per *target
//!   config*). This lets mirage run the existing corpus unmodified.
//! * A native, pure-text [TOML] format that additionally supports
//!   [Common Expression Language] (CEL) expressions for **input
//!   generation** and **output validation**, so contributors can describe
//!   parametric tests without checking in large tensors.
//!
//! The pipeline mirrors the upstream `iree_corpus.py` runner:
//!
//! 1. `iree-compile` each `.mlir` source into a `.vmfb` (cached by content
//!    hash), on the host.
//! 2. `iree-run-module`, wrapped by `mirage run --profile <scenario> --`,
//!    so the emulator's environment contract is injected around the run.
//! 3. Validate the produced `.npy` outputs against expectations.
//!
//! [`ROCm/rocjitsu-test-corpus`]: https://github.com/ROCm/rocjitsu-test-corpus
//! [TOML]: https://toml.io
//! [Common Expression Language]: https://github.com/google/cel-spec

pub mod cel;
pub mod error;
pub mod generate;
pub mod loader;
pub mod model;
pub mod report;
pub mod runner;
pub mod scenario;
pub mod tensor;
pub mod validate;

pub use error::{CorpusError, Result};
pub use loader::{discover_cases, is_case_path, load_case, load_target_config};
pub use model::{Case, Check, InputSpec, OutputSpec, Scenario, TargetConfig};
pub use report::{CaseOutcome, CaseStatus, RunReport};
pub use runner::{RunOptions, run_case};
pub use scenario::{ScenarioKind, builtin_scenarios};
pub use tensor::Tensor;
