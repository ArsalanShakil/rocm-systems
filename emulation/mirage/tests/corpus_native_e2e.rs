//! End-to-end tests for the native `mirage corpus` pipeline.
//!
//! These drive the unified `mirage` binary against the bundled
//! `examples/corpus-demo` fixture, which ships *stub* `iree-compile` /
//! `iree-run-module` tools so the full compile → run → validate pipeline
//! executes hermetically — no real IREE, no GPU, no network.
//!
//! The run-pipeline assertions require `python3` (used by the stub
//! `iree-run-module`); when it is unavailable they are skipped.

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

fn mirage_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mirage"))
}

fn demo_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/corpus-demo")
}

/// Prepend the demo's stub-tools dir to `PATH` for `cmd`.
fn with_stub_path(cmd: &mut Command) {
    let tools = demo_dir().join("tools");
    let existing = std::env::var("PATH").unwrap_or_default();
    cmd.env("PATH", format!("{}:{existing}", tools.display()));
}

fn have_python3() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn scenarios_lists_the_three_builtins() {
    let mut cmd = Command::new(mirage_bin());
    cmd.args(["corpus", "scenarios"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("rocjitsu"))
        .stdout(predicate::str::contains("hotswap"))
        .stdout(predicate::str::contains("native"));
}

#[test]
fn scenarios_json_is_machine_readable() {
    let out = Command::new(mirage_bin())
        .args(["--json", "corpus", "scenarios"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let arr = v.as_array().expect("scenarios is an array");
    assert_eq!(arr.len(), 3);
    assert!(arr.iter().any(|s| s["name"] == "native"));
}

#[test]
fn list_discovers_the_demo_case() {
    let mut cmd = Command::new(mirage_bin());
    cmd.args(["corpus", "list", "--root"]).arg(demo_dir());
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("relu_f32"));
}

#[test]
fn show_emits_case_json() {
    let out = Command::new(mirage_bin())
        .args(["corpus", "show", "--root"])
        .arg(demo_dir())
        .arg("relu_f32")
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["name"], "relu_f32");
    assert_eq!(v["function"], "main");
}

#[test]
fn run_native_passes_with_stub_tools() {
    if !have_python3() {
        eprintln!("skipping: python3 not available for stub iree-run-module");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let mut cmd = Command::new(mirage_bin());
    with_stub_path(&mut cmd);
    cmd.args(["corpus", "run", "--root"])
        .arg(demo_dir())
        .args(["--scenario", "native", "--no-wrapper", "--artifact-dir"])
        .arg(tmp.path().join("artifacts"))
        .args(["--out-dir"])
        .arg(tmp.path().join("results"));
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("1 passed, 0 failed"));

    // The CSV report is written to the out-dir.
    let csv = std::fs::read_to_string(tmp.path().join("results/results.csv")).unwrap();
    assert!(csv.contains("relu_f32"));
    assert!(csv.contains("native"));
    assert!(csv.contains("pass"));
}

#[test]
fn run_native_json_report_is_parseable() {
    if !have_python3() {
        eprintln!("skipping: python3 not available for stub iree-run-module");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let mut cmd = Command::new(mirage_bin());
    with_stub_path(&mut cmd);
    cmd.args(["--json", "corpus", "run", "--root"])
        .arg(demo_dir())
        .args(["--scenario", "native", "--no-wrapper", "--artifact-dir"])
        .arg(tmp.path().join("artifacts"));
    let out = cmd.output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let outcomes = v["outcomes"].as_array().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0]["case"], "relu_f32");
    assert_eq!(outcomes[0]["status"], "pass");
}
