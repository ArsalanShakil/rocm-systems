//! Loading and discovery of cases and target configs from disk.
//!
//! Both the upstream JSON layout and the native TOML layout are accepted.
//! A file is treated as a *case* when it is a `.json`/`.toml` document that
//! is not under a `configs/` or `schemas/` directory.

use std::path::{Path, PathBuf};

use crate::error::{CorpusError, Result};
use crate::model::{Case, TargetConfig};

/// Whether `path` looks like a corpus case document (by location + suffix).
pub fn is_case_path(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "json" && ext != "toml" {
        return false;
    }
    // Exclude config/schema documents anywhere in the path.
    !path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some("configs") | Some("schemas")
        )
    })
}

/// Read a document as raw text.
fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|source| CorpusError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Load a single case from a `.json` or `.toml` document.
pub fn load_case(path: &Path) -> Result<Case> {
    let text = read(path)?;
    let mut case: Case = parse(path, &text)?;
    case.path = path.to_path_buf();
    finalize_case(path, &mut case)?;
    Ok(case)
}

/// Load a target config from a `.json` or `.toml` document.
pub fn load_target_config(path: &Path) -> Result<TargetConfig> {
    let text = read(path)?;
    let mut cfg: TargetConfig = parse(path, &text)?;
    cfg.path = path.to_path_buf();
    Ok(cfg)
}

fn parse<T: serde::de::DeserializeOwned>(path: &Path, text: &str) -> Result<T> {
    let is_toml = path.extension().and_then(|e| e.to_str()) == Some("toml");
    if is_toml {
        toml::from_str(text).map_err(|source| CorpusError::Toml {
            path: path.to_path_buf(),
            source,
        })
    } else {
        serde_json::from_str(text).map_err(|source| CorpusError::Json {
            path: path.to_path_buf(),
            source,
        })
    }
}

/// Apply post-load validation and defaults, mirroring `iree_corpus.py`.
fn finalize_case(path: &Path, case: &mut Case) -> Result<()> {
    if case.kind != "run_module" {
        return Err(CorpusError::invalid(
            path,
            format!("unsupported kind {:?} (only run_module)", case.kind),
        ));
    }
    if case.sources.is_empty() {
        return Err(CorpusError::invalid(path, "must list at least one source"));
    }
    if case.vmfb_names.is_empty() {
        case.vmfb_names = case
            .sources
            .iter()
            .map(|s| {
                let stem = Path::new(s)
                    .file_stem()
                    .and_then(|x| x.to_str())
                    .unwrap_or("module");
                format!("{stem}.vmfb")
            })
            .collect();
    }
    if case.vmfb_names.len() != case.sources.len() {
        return Err(CorpusError::invalid(
            path,
            "must have one vmfb_name per source",
        ));
    }
    Ok(())
}

/// Recursively discover all case documents beneath `root`.
///
/// Files that fail to parse are surfaced as errors; callers may choose to
/// log-and-skip them. Results are sorted by path for stable ordering.
pub fn discover_cases(root: &Path) -> Result<Vec<Case>> {
    let mut paths = Vec::new();
    walk(root, &mut paths)?;
    paths.sort();
    let mut cases = Vec::new();
    for p in paths {
        if is_case_path(&p) {
            cases.push(load_case(&p)?);
        }
    }
    Ok(cases)
}

/// Recursively discover case *paths* (without loading) beneath `root`.
pub fn discover_case_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    walk(root, &mut paths)?;
    paths.retain(|p| is_case_path(p));
    paths.sort();
    Ok(paths)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    if dir.is_file() {
        out.push(dir.to_path_buf());
        return Ok(());
    }
    let entries = std::fs::read_dir(dir).map_err(|source| CorpusError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| CorpusError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            // Skip hidden directories (e.g. `.git`, and the demo's
            // `.artifacts` / `.results` output dirs) so generated files
            // never masquerade as corpus cases.
            let hidden = entry
                .file_name()
                .to_str()
                .map(|n| n.starts_with('.'))
                .unwrap_or(false);
            if !hidden {
                walk(&path, out)?;
            }
        } else {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, text: &str) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn loads_upstream_json_case() {
        let d = tempfile::tempdir().unwrap();
        let p = write(
            d.path(),
            "e2e/case.json",
            r#"{"name":"abs","kind":"run_module","sources":["abs.mlir"],"function":"abs"}"#,
        );
        let case = load_case(&p).unwrap();
        assert_eq!(case.name, "abs");
        assert_eq!(case.vmfb_names, vec!["abs.vmfb"]);
    }

    #[test]
    fn loads_native_toml_case() {
        let d = tempfile::tempdir().unwrap();
        let p = write(
            d.path(),
            "mm/case.toml",
            r#"
name = "mm"
sources = ["mm.mlir"]
function = "matmul"
validate = "allclose"

[params]
n = 4

[[inputs]]
dtype = "float32"
formula = "{'dist': 'uniform', 'shape': [n, n], 'low': 0.0, 'high': 1.0}"
"#,
        );
        let case = load_case(&p).unwrap();
        assert_eq!(case.name, "mm");
        assert_eq!(case.kind, "run_module");
        assert_eq!(case.validate_exprs(), vec!["allclose".to_string()]);
        assert_eq!(case.inputs.len(), 1);
    }

    #[test]
    fn rejects_bad_kind() {
        let d = tempfile::tempdir().unwrap();
        let p = write(
            d.path(),
            "bad/case.json",
            r#"{"name":"x","kind":"trace","sources":["a.mlir"],"function":"f"}"#,
        );
        assert!(load_case(&p).is_err());
    }

    #[test]
    fn is_case_path_excludes_configs_and_schemas() {
        assert!(is_case_path(Path::new("corpus/iree/e2e/a.json")));
        assert!(is_case_path(Path::new("corpus/mm/case.toml")));
        assert!(!is_case_path(Path::new("corpus/iree/configs/gfx1250.json")));
        assert!(!is_case_path(Path::new("corpus/iree/schemas/case.json")));
        assert!(!is_case_path(Path::new("corpus/iree/e2e/a.mlir")));
    }

    #[test]
    fn discover_finds_cases_and_skips_configs() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "iree/e2e/a.json",
            r#"{"name":"a","kind":"run_module","sources":["a.mlir"],"function":"f"}"#,
        );
        write(
            d.path(),
            "iree/configs/gfx.json",
            r#"{"config_name":"gfx","iree_compile_flags":[],"iree_run_module_flags":[]}"#,
        );
        let cases = discover_cases(d.path()).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].name, "a");
    }
}
