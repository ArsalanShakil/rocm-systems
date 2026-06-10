//! The end-to-end run pipeline: compile, run (through a mirage scenario),
//! and validate a single case.
//!
//! The shape mirrors `iree_corpus.py`: `iree-compile` produces a cached
//! `.vmfb` per source, then `iree-run-module` is invoked — wrapped by
//! `mirage run --profile <scenario> --` — and the observed `.npy` outputs
//! are validated.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::error::{CorpusError, Result};
use crate::generate::materialize_inputs;
use crate::model::{Case, TargetConfig};
use crate::report::{CaseOutcome, CaseStatus};
use crate::scenario::ScenarioKind;
use crate::validate::{expected_tensors, validate};

/// Options controlling how a case is compiled and run.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// The `mirage` binary used to wrap `iree-run-module`. When `None`
    /// (and no explicit `run_wrapper`), the run is *not* wrapped — useful
    /// for tests with stub tools.
    pub mirage_bin: Option<PathBuf>,
    /// Where compiled modules, inputs, outputs, and logs are written.
    pub artifact_dir: PathBuf,
    /// Compile every case but never run it.
    pub compile_only: bool,
    /// Explicit run-wrapper tokens (overrides the scenario default).
    pub run_wrapper: Option<Vec<String>>,
    /// The `iree-compile` executable name / path.
    pub iree_compile: String,
    /// The `iree-run-module` executable name / path.
    pub iree_run_module: String,
    /// Whether to ensure the scenario's mirage profile exists before
    /// running (requires `mirage_bin`).
    pub ensure_profile: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            mirage_bin: None,
            artifact_dir: PathBuf::from(".corpus-artifacts"),
            compile_only: false,
            run_wrapper: None,
            iree_compile: "iree-compile".to_string(),
            iree_run_module: "iree-run-module".to_string(),
            ensure_profile: true,
        }
    }
}

/// Run a single case under one target config and scenario.
///
/// Never returns `Err` for an expected outcome: compile/run/validation
/// failures and missing prerequisites are encoded in the [`CaseOutcome`]
/// status (so a batch run can keep going).
pub fn run_case(
    case: &Case,
    config: &TargetConfig,
    scenario: ScenarioKind,
    opts: &RunOptions,
) -> CaseOutcome {
    let started = Instant::now();
    let outcome = |status: CaseStatus, returncode: i32, message: String| CaseOutcome {
        case: case.name.clone(),
        scenario: scenario.name().to_string(),
        config: config.config_name.clone(),
        status,
        elapsed_s: started.elapsed().as_secs_f64(),
        returncode,
        message,
    };

    // Cases explicitly skipped for this config are not collected upstream;
    // we surface them as a skip so the matrix stays complete.
    if config.skip_compile_tests.contains(&case.name) {
        return outcome(CaseStatus::Skip, 0, "skip_compile_tests".into());
    }

    let expect_compile_fail = config.expected_compile_failures.contains(&case.name);
    let expect_run_fail = config.expected_run_failures.contains(&case.name);

    match try_run(case, config, scenario, opts) {
        Ok(RunVerdict::Compiled) => {
            if expect_compile_fail {
                outcome(CaseStatus::Xpass, 0, "expected compile failure but compiled".into())
            } else {
                outcome(CaseStatus::Pass, 0, "compiled".into())
            }
        }
        Ok(RunVerdict::Ran) => {
            if expect_run_fail {
                outcome(CaseStatus::Xpass, 0, "expected run failure but passed".into())
            } else {
                outcome(CaseStatus::Pass, 0, String::new())
            }
        }
        Err(CorpusError::ToolMissing(t)) => {
            outcome(CaseStatus::Skip, 77, format!("tool not found: {t}"))
        }
        Err(CorpusError::ScenarioUnavailable(m)) => outcome(CaseStatus::Skip, 77, m),
        Err(CorpusError::Tool { phase, code, log, .. }) if phase == "compile" => {
            if expect_compile_fail {
                outcome(CaseStatus::Xfail, code, "expected compile failure".into())
            } else {
                outcome(CaseStatus::Fail, code, truncate(&log))
            }
        }
        Err(e) => {
            // Run or validation failure.
            if expect_run_fail {
                outcome(CaseStatus::Xfail, 1, "expected run failure".into())
            } else {
                outcome(CaseStatus::Fail, 1, truncate(&e.to_string()))
            }
        }
    }
}

enum RunVerdict {
    Compiled,
    Ran,
}

fn try_run(
    case: &Case,
    config: &TargetConfig,
    scenario: ScenarioKind,
    opts: &RunOptions,
) -> Result<RunVerdict> {
    // Absolutize the artifact dir up front: the compile/run steps execute
    // with `cwd` set to the case/source directory, so every module, input,
    // and output path we hand to the tools must be absolute to resolve
    // regardless of that working directory.
    let opts_owned = RunOptions {
        artifact_dir: absolutize(&opts.artifact_dir),
        ..opts.clone()
    };
    let opts = &opts_owned;

    let case_dir = case.dir();
    let run_dir = run_dir(&opts.artifact_dir, config, case);
    std::fs::create_dir_all(&run_dir).map_err(|source| CorpusError::Io {
        path: run_dir.clone(),
        source,
    })?;

    // 1. Compile each source into a cached module.
    let mut modules = Vec::with_capacity(case.sources.len());
    for (source, vmfb) in case.sources.iter().zip(&case.vmfb_names) {
        modules.push(compile_source(
            &case_dir.join(source),
            vmfb,
            case,
            config,
            opts,
        )?);
    }

    let compile_only = opts.compile_only
        || case.compile_only
        || config.skip_run_tests.contains(&case.name);
    if compile_only {
        return Ok(RunVerdict::Compiled);
    }

    // 2. Ensure the scenario profile + build the run wrapper.
    let wrapper = resolve_wrapper(scenario, opts)?;

    // 3. Materialize inputs and prepare output paths.
    let inputs = materialize_inputs(case, &run_dir)?;
    let output_paths: Vec<PathBuf> = case
        .outputs
        .iter()
        .map(|o| run_dir.join(o))
        .collect();
    for o in &output_paths {
        if let Some(parent) = o.parent() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    // 4. Build and run `iree-run-module` (wrapped).
    let mut argv: Vec<String> = wrapper;
    argv.push(opts.iree_run_module.clone());
    argv.extend(config.iree_run_module_flags.iter().cloned());
    argv.extend(case.run_flags.iter().cloned());
    for m in &modules {
        argv.push(format!("--module={}", m.display()));
    }
    argv.push(format!("--function={}", case.function));
    for inp in &inputs {
        argv.push(format!("--input=@{}", inp.path.display()));
    }
    for out in &output_paths {
        argv.push(format!("--output=@{}", out.display()));
    }
    run_command(&argv, &case_dir, &run_dir.join("run.log"), "run")?;

    // 5. Validate.
    let observed: Vec<crate::tensor::Tensor> = output_paths
        .iter()
        .map(|p| crate::tensor::Tensor::read_npy(p))
        .collect::<Result<_>>()?;
    let input_tensors: Vec<crate::tensor::Tensor> =
        inputs.into_iter().map(|i| i.tensor).collect();
    let expected = expected_tensors(case, &input_tensors)?;
    validate(case, &observed, &expected)?;

    Ok(RunVerdict::Ran)
}

/// Resolve the run-wrapper tokens for a scenario, ensuring its profile.
fn resolve_wrapper(scenario: ScenarioKind, opts: &RunOptions) -> Result<Vec<String>> {
    if let Some(w) = &opts.run_wrapper {
        return Ok(w.clone());
    }
    let Some(mirage) = &opts.mirage_bin else {
        // No wrapping: run the tool directly (test / bare-host mode).
        return Ok(Vec::new());
    };
    let profile = scenario.profile_name();
    if opts.ensure_profile {
        ensure_profile(mirage, scenario)?;
    }
    Ok(vec![
        mirage.display().to_string(),
        "run".into(),
        "--profile".into(),
        profile,
        "--".into(),
    ])
}

/// Ensure the mirage profile for `scenario` exists, importing it if needed.
fn ensure_profile(mirage: &Path, scenario: ScenarioKind) -> Result<()> {
    let profile = scenario.profile_name();
    let shown = Command::new(mirage)
        .args(["profile", "show", &profile])
        .output();
    if matches!(&shown, Ok(o) if o.status.success()) {
        return Ok(());
    }
    // Import the generated profile via stdin.
    use std::io::Write;
    let json = scenario.profile_json()?;
    let mut child = Command::new(mirage)
        .args(["profile", "import", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| map_spawn_err(e, &mirage.display().to_string()))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(json.as_bytes()).ok();
    }
    let out = child
        .wait_with_output()
        .map_err(|e| CorpusError::other(format!("mirage import wait: {e}")))?;
    if !out.status.success() {
        return Err(CorpusError::ScenarioUnavailable(format!(
            "could not create profile '{profile}' (is the '{}' emulator installed?): {}",
            scenario.emulator(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(())
}

/// Compile one source into a content-addressed `.vmfb`, reusing the cache.
fn compile_source(
    source: &Path,
    vmfb_name: &str,
    case: &Case,
    config: &TargetConfig,
    opts: &RunOptions,
) -> Result<PathBuf> {
    if !source.exists() {
        return Err(CorpusError::invalid(
            source,
            "source file does not exist".to_string(),
        ));
    }
    let mut flags = config.iree_compile_flags.clone();
    flags.extend(case.compile_flags.iter().cloned());

    let digest = compile_digest(source, &config.config_name, &flags);
    let output = opts
        .artifact_dir
        .join(&config.config_name)
        .join("compile-cache")
        .join(&digest)
        .join(vmfb_name);

    let fresh = output.exists()
        && file_mtime(&output) >= file_mtime(source);
    if fresh {
        return Ok(output);
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|source| CorpusError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let mut argv = vec![opts.iree_compile.clone(), source.display().to_string()];
    argv.extend(flags);
    argv.push("-o".into());
    argv.push(output.display().to_string());
    let log = output.with_extension("compile.log");
    run_command(
        &argv,
        source.parent().unwrap_or(Path::new(".")),
        &log,
        "compile",
    )?;
    Ok(output)
}

fn compile_digest(source: &Path, config_name: &str, flags: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.display().to_string().as_bytes());
    hasher.update([0]);
    hasher.update(config_name.as_bytes());
    hasher.update([0]);
    for f in flags {
        hasher.update(f.as_bytes());
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    hex16(&digest)
}

fn hex16(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(16);
    for b in bytes.iter().take(8) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn file_mtime(path: &Path) -> std::time::SystemTime {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::UNIX_EPOCH)
}

/// Make a path absolute (joining the process CWD if it is relative),
/// without requiring the path to exist (unlike `canonicalize`).
fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

/// Run a command, writing a log file, mapping failures to [`CorpusError`].
fn run_command(argv: &[String], cwd: &Path, log_path: &Path, phase: &str) -> Result<()> {
    if argv.is_empty() {
        return Err(CorpusError::other("empty command"));
    }
    let tool = &argv[0];
    let output = Command::new(tool)
        .args(&argv[1..])
        .current_dir(cwd)
        .output()
        .map_err(|e| map_spawn_err(e, tool))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = std::fs::write(
        log_path,
        format!(
            "$ {}\ncwd: {}\nreturncode: {}\n\nstdout:\n{}\nstderr:\n{}\n",
            argv.join(" "),
            cwd.display(),
            output.status.code().unwrap_or(-1),
            stdout,
            stderr
        ),
    );

    if !output.status.success() {
        return Err(CorpusError::Tool {
            phase: phase.to_string(),
            tool: tool.clone(),
            code: output.status.code().unwrap_or(-1),
            log: format!("stdout:\n{stdout}\nstderr:\n{stderr}"),
        });
    }
    Ok(())
}

fn map_spawn_err(e: std::io::Error, tool: &str) -> CorpusError {
    if e.kind() == std::io::ErrorKind::NotFound {
        CorpusError::ToolMissing(tool.to_string())
    } else {
        CorpusError::other(format!("failed to spawn {tool}: {e}"))
    }
}

fn run_dir(artifact_dir: &Path, config: &TargetConfig, case: &Case) -> PathBuf {
    artifact_dir
        .join(&config.config_name)
        .join("runs")
        .join(&case.name)
}

fn truncate(s: &str) -> String {
    const MAX: usize = 2000;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}…", &s[..MAX])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_digest_is_stable_and_flag_sensitive() {
        let s = Path::new("/c/a.mlir");
        let a = compile_digest(s, "gfx", &["-x".into()]);
        let b = compile_digest(s, "gfx", &["-x".into()]);
        let c = compile_digest(s, "gfx", &["-y".into()]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16);
    }
}
