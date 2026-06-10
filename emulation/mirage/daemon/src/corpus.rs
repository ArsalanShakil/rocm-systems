//! REST handlers for the `/api/corpus` surface: browse scenarios,
//! discover cases, and run them across scenarios.
//!
//! These delegate to [`mirage_corpus`] and reuse the same discovery and
//! run pipeline that the `mirage corpus` CLI drives, so the dashboard and
//! the CLI stay in lock-step.

use std::path::PathBuf;

use axum::Json;
use axum::Router;
use axum::extract::Query;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use mirage_corpus::loader;
use mirage_corpus::model::TargetConfig;
use mirage_corpus::report::RunReport;
use mirage_corpus::runner::{RunOptions, run_case};
use mirage_corpus::scenario::{ScenarioKind, builtin_scenarios};
use serde::Deserialize;

use crate::state::AppState;
use std::sync::Arc;

/// Mount the corpus routes (the caller nests this under `/api`).
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/corpus/scenarios", get(list_scenarios))
        .route("/corpus/cases", get(list_cases))
        .route("/corpus/run", post(run))
}

struct CorpusError(StatusCode, String);

impl IntoResponse for CorpusError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({"error": self.1}));
        (self.0, body).into_response()
    }
}

impl From<mirage_corpus::error::CorpusError> for CorpusError {
    fn from(e: mirage_corpus::error::CorpusError) -> Self {
        CorpusError(StatusCode::BAD_REQUEST, e.to_string())
    }
}

async fn list_scenarios() -> Json<Vec<serde_json::Value>> {
    let out = builtin_scenarios()
        .iter()
        .map(mirage_ctl::corpus::scenario_json)
        .collect();
    Json(out)
}

#[derive(Deserialize)]
struct RootQuery {
    root: String,
}

async fn list_cases(Query(q): Query<RootQuery>) -> Result<Json<serde_json::Value>, CorpusError> {
    let cases = loader::discover_cases(&PathBuf::from(&q.root))?;
    Ok(Json(serde_json::to_value(&cases).map_err(|e| {
        CorpusError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?))
}

#[derive(Deserialize)]
struct RunBody {
    root: String,
    #[serde(default)]
    configs: Vec<String>,
    #[serde(default)]
    cases: Vec<String>,
    #[serde(default)]
    scenarios: Vec<String>,
    #[serde(default)]
    compile_only: bool,
    #[serde(default)]
    no_wrapper: bool,
    artifact_dir: Option<String>,
}

async fn run(Json(body): Json<RunBody>) -> Result<Json<RunReport>, CorpusError> {
    let report = tokio::task::spawn_blocking(move || run_blocking(body))
        .await
        .map_err(|e| CorpusError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(report))
}

fn run_blocking(body: RunBody) -> Result<RunReport, CorpusError> {
    let mut cases = loader::discover_cases(&PathBuf::from(&body.root))?;
    if !body.cases.is_empty() {
        cases.retain(|c| body.cases.contains(&c.name));
    }
    if cases.is_empty() {
        return Err(CorpusError(
            StatusCode::NOT_FOUND,
            format!("no matching cases under {}", body.root),
        ));
    }

    let configs = if body.configs.is_empty() {
        vec![default_config()]
    } else {
        let mut v = Vec::new();
        for p in &body.configs {
            v.push(loader::load_target_config(&PathBuf::from(p))?);
        }
        v
    };

    let kinds = if body.scenarios.is_empty() {
        vec![
            ScenarioKind::Rocjitsu,
            ScenarioKind::Hotswap,
            ScenarioKind::Native,
        ]
    } else {
        let mut v = Vec::new();
        for n in &body.scenarios {
            v.push(
                ScenarioKind::parse(n)
                    .map_err(|e| CorpusError(StatusCode::BAD_REQUEST, e.to_string()))?,
            );
        }
        v
    };

    let mirage_bin = if body.no_wrapper {
        None
    } else {
        std::env::current_exe().ok()
    };
    let artifact_dir = body
        .artifact_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".corpus-artifacts"));

    let mut report = RunReport::default();
    for case in &cases {
        for config in &configs {
            for &kind in &kinds {
                let opts = RunOptions {
                    mirage_bin: mirage_bin.clone(),
                    artifact_dir: artifact_dir.clone(),
                    compile_only: body.compile_only,
                    run_wrapper: None,
                    iree_compile: "iree-compile".into(),
                    iree_run_module: "iree-run-module".into(),
                    ensure_profile: true,
                };
                report.push(run_case(case, config, kind, &opts));
            }
        }
    }
    Ok(report)
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
