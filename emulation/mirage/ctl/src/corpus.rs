//! The `mirage corpus` subcommand: browse and run rocjitsu test-corpus
//! cases under one or more emulator *scenarios*.
//!
//! This is a thin CLI over [`mirage_corpus`]. It discovers cases (upstream
//! JSON or native TOML), resolves the scenarios to run them under, drives
//! the compile/run/validate pipeline, and prints a results matrix.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};
use mirage_corpus::loader;
use mirage_corpus::model::{Case, Scenario, TargetConfig};
use mirage_corpus::report::{CaseStatus, RunReport};
use mirage_corpus::runner::{RunOptions, run_case};
use mirage_corpus::scenario::{ScenarioKind, builtin_scenarios};

/// `mirage corpus` subcommands.
#[derive(Subcommand, Debug)]
pub enum CorpusCmd {
    /// List the available run scenarios and their install status.
    Scenarios,

    /// List the cases discovered under a corpus root.
    List(ListArgs),

    /// Show a single case document.
    Show(ShowArgs),

    /// Compile, run, and validate cases across scenarios.
    Run(RunArgs),

    /// Benchmark cases across scenarios (wall-clock per scenario).
    Bench(RunArgs),
}

/// Arguments shared by `list`/`show`.
#[derive(Args, Debug)]
pub struct ListArgs {
    /// Corpus root directory to discover cases under.
    #[arg(long)]
    root: PathBuf,
}

/// Arguments for `show`.
#[derive(Args, Debug)]
pub struct ShowArgs {
    /// Corpus root directory to discover cases under.
    #[arg(long)]
    root: PathBuf,
    /// The case name to show.
    name: String,
}

/// Arguments for `run`/`bench`.
#[derive(Args, Debug)]
pub struct RunArgs {
    /// Corpus root directory to discover cases under.
    #[arg(long)]
    root: PathBuf,

    /// Target config file(s) (compile/run flags + skip/xfail lists).
    /// When omitted a permissive default config is synthesized.
    #[arg(long = "config", value_name = "PATH")]
    configs: Vec<PathBuf>,

    /// Only run cases with these names (repeatable). Default: all.
    #[arg(long = "case", short = 'c', value_name = "NAME")]
    cases: Vec<String>,

    /// Scenarios to run (repeatable): rocjitsu, hotswap, native.
    /// Default: all built-in scenarios (unavailable ones are skipped).
    #[arg(long = "scenario", short = 's', value_name = "NAME")]
    scenarios: Vec<String>,

    /// Directory for compiled modules, inputs, outputs, and logs.
    #[arg(long, default_value = ".corpus-artifacts")]
    artifact_dir: PathBuf,

    /// Write `results.csv` and `results.json` to this directory.
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Compile cases but never run them.
    #[arg(long)]
    compile_only: bool,

    /// Run the IREE tools directly, without wrapping them in `mirage run`.
    #[arg(long)]
    no_wrapper: bool,

    /// Override the `iree-compile` executable.
    #[arg(long, default_value = "iree-compile")]
    iree_compile: String,

    /// Override the `iree-run-module` executable.
    #[arg(long, default_value = "iree-run-module")]
    iree_run_module: String,

    /// Path to the `mirage` binary used to wrap runs (default: this one).
    #[arg(long)]
    mirage_bin: Option<PathBuf>,
}

/// Dispatch a `mirage corpus` subcommand.
pub fn corpus_cmd(cmd: CorpusCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        CorpusCmd::Scenarios => scenarios_cmd(json),
        CorpusCmd::List(a) => list_cmd(a, json),
        CorpusCmd::Show(a) => show_cmd(a, json),
        CorpusCmd::Run(a) => run_cmd(a, json, false),
        CorpusCmd::Bench(a) => run_cmd(a, json, true),
    }
}

fn scenarios_cmd(json: bool) -> anyhow::Result<ExitCode> {
    let scenarios = builtin_scenarios();
    if json {
        let enriched: Vec<_> = scenarios
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "emulator": s.emulator,
                    "agent": s.agent,
                    "profile": s.profile,
                    "description": s.description,
                    "installed": emulator_installed(&s.emulator),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&enriched)?);
        return Ok(ExitCode::from(0));
    }
    println!("{:<10} {:<10} {:<8} {:<10} DESCRIPTION", "SCENARIO", "EMULATOR", "AGENT", "INSTALLED");
    for s in &scenarios {
        println!(
            "{:<10} {:<10} {:<8} {:<10} {}",
            s.name,
            s.emulator,
            s.agent,
            if emulator_installed(&s.emulator) { "yes" } else { "no" },
            s.description
        );
    }
    Ok(ExitCode::from(0))
}

fn list_cmd(a: ListArgs, json: bool) -> anyhow::Result<ExitCode> {
    let cases = loader::discover_cases(&a.root)?;
    if json {
        let rows: Vec<_> = cases
            .iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "kind": c.kind,
                    "sources": c.sources,
                    "function": c.function,
                    "compile_only": c.compile_only,
                    "path": c.path,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(ExitCode::from(0));
    }
    if cases.is_empty() {
        eprintln!("(no cases found under {})", a.root.display());
    }
    println!("{:<32} {:<12} FUNCTION", "NAME", "KIND");
    for c in &cases {
        println!("{:<32} {:<12} {}", c.name, c.kind, c.function);
    }
    Ok(ExitCode::from(0))
}

fn show_cmd(a: ShowArgs, _json: bool) -> anyhow::Result<ExitCode> {
    let cases = loader::discover_cases(&a.root)?;
    let Some(case) = cases.into_iter().find(|c| c.name == a.name) else {
        anyhow::bail!("case not found: {}", a.name);
    };
    println!("{}", serde_json::to_string_pretty(&case)?);
    Ok(ExitCode::from(0))
}

fn run_cmd(a: RunArgs, json: bool, bench: bool) -> anyhow::Result<ExitCode> {
    let cases = filter_cases(loader::discover_cases(&a.root)?, &a.cases);
    if cases.is_empty() {
        anyhow::bail!("no matching cases found under {}", a.root.display());
    }
    let configs = resolve_configs(&a.configs)?;
    let scenario_kinds = resolve_scenarios(&a.scenarios)?;

    let mirage_bin = if a.no_wrapper {
        None
    } else {
        Some(
            a.mirage_bin
                .clone()
                .or_else(|| std::env::current_exe().ok())
                .unwrap_or_else(|| PathBuf::from("mirage")),
        )
    };

    let mut report = RunReport::default();
    for case in &cases {
        for config in &configs {
            for &kind in &scenario_kinds {
                let opts = RunOptions {
                    mirage_bin: mirage_bin.clone(),
                    artifact_dir: a.artifact_dir.clone(),
                    compile_only: a.compile_only,
                    run_wrapper: None,
                    iree_compile: a.iree_compile.clone(),
                    iree_run_module: a.iree_run_module.clone(),
                    ensure_profile: true,
                };
                report.push(run_case(case, config, kind, &opts));
            }
        }
    }

    if let Some(out_dir) = &a.out_dir {
        std::fs::create_dir_all(out_dir)?;
        std::fs::write(out_dir.join("results.csv"), report.to_csv())?;
        std::fs::write(
            out_dir.join("results.json"),
            serde_json::to_string_pretty(&report)?,
        )?;
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_matrix(&report, &cases, &scenario_kinds, bench);
        println!("\n{}", report.summary());
    }

    Ok(if report.any_failures() {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    })
}

fn filter_cases(cases: Vec<Case>, names: &[String]) -> Vec<Case> {
    if names.is_empty() {
        return cases;
    }
    cases
        .into_iter()
        .filter(|c| names.contains(&c.name))
        .collect()
}

fn resolve_configs(paths: &[PathBuf]) -> anyhow::Result<Vec<TargetConfig>> {
    if paths.is_empty() {
        return Ok(vec![default_config()]);
    }
    let mut configs = Vec::new();
    for p in paths {
        configs.push(loader::load_target_config(p)?);
    }
    Ok(configs)
}

fn default_config() -> TargetConfig {
    TargetConfig {
        config_name: "default".into(),
        iree_compile_flags: Vec::new(),
        iree_run_module_flags: Vec::new(),
        iree_run_module_wrapper: Vec::new(),
        skip_compile_tests: Vec::new(),
        expected_compile_failures: Vec::new(),
        skip_run_tests: Vec::new(),
        expected_run_failures: Vec::new(),
        path: PathBuf::new(),
    }
}

fn resolve_scenarios(names: &[String]) -> anyhow::Result<Vec<ScenarioKind>> {
    if names.is_empty() {
        return Ok(vec![
            ScenarioKind::Rocjitsu,
            ScenarioKind::Hotswap,
            ScenarioKind::Native,
        ]);
    }
    let mut kinds = Vec::new();
    for n in names {
        kinds.push(ScenarioKind::parse(n).map_err(|e| anyhow::anyhow!("{e}"))?);
    }
    Ok(kinds)
}

fn emulator_installed(emulator: &str) -> bool {
    crate::find_emulator(emulator)
        .map(|e| e.installed)
        .unwrap_or(false)
}

fn print_matrix(report: &RunReport, cases: &[Case], scenarios: &[ScenarioKind], bench: bool) {
    let names: Vec<&str> = scenarios.iter().map(|s| s.name()).collect();
    print!("{:<32}", "CASE");
    for n in &names {
        print!(" {:<12}", n);
    }
    println!();
    for case in cases {
        print!("{:<32}", case.name);
        for kind in scenarios {
            let cell = report
                .outcomes
                .iter()
                .find(|o| o.case == case.name && o.scenario == kind.name());
            let text = match cell {
                Some(o) if bench && o.status == CaseStatus::Pass => {
                    format!("{:.3}s", o.elapsed_s)
                }
                Some(o) => o.status.label().to_string(),
                None => "-".to_string(),
            };
            print!(" {text:<12}");
        }
        println!();
    }
}

/// Build the daemon-facing JSON description of a scenario (re-used by the API).
pub fn scenario_json(s: &Scenario) -> serde_json::Value {
    serde_json::json!({
        "name": s.name,
        "emulator": s.emulator,
        "agent": s.agent,
        "profile": s.profile,
        "description": s.description,
        "installed": emulator_installed(&s.emulator),
    })
}
