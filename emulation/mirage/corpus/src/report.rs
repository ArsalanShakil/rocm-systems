//! Run results and reporting (CSV + JSON + summary).

use serde::{Deserialize, Serialize};

/// Terminal status of a single case run under one scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaseStatus {
    /// Compiled, ran, and validated successfully.
    Pass,
    /// Compiled / ran / validated but produced a wrong result or error.
    Fail,
    /// Could not run (missing tool, emulator not installed, etc.).
    Skip,
    /// Expected to fail and did (counts as success).
    Xfail,
    /// Expected to fail but passed (counts as a failure).
    Xpass,
}

impl CaseStatus {
    /// The lowercase label used in CSV output.
    pub fn label(&self) -> &'static str {
        match self {
            CaseStatus::Pass => "pass",
            CaseStatus::Fail => "fail",
            CaseStatus::Skip => "skip",
            CaseStatus::Xfail => "xfail",
            CaseStatus::Xpass => "xpass",
        }
    }

    /// Whether this status represents an overall failure.
    pub fn is_failure(&self) -> bool {
        matches!(self, CaseStatus::Fail | CaseStatus::Xpass)
    }
}

/// The outcome of running one case under one scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseOutcome {
    /// The case name.
    pub case: String,
    /// The scenario name (`rocjitsu`, `hotswap`, `native`).
    pub scenario: String,
    /// The target config name.
    pub config: String,
    /// Terminal status.
    pub status: CaseStatus,
    /// Wall-clock seconds the run took.
    pub elapsed_s: f64,
    /// Process return code of the run step (0 when not applicable).
    pub returncode: i32,
    /// A short human-readable message (failure reason / skip reason).
    pub message: String,
}

/// A full report over many cases and scenarios.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunReport {
    /// All individual outcomes.
    pub outcomes: Vec<CaseOutcome>,
}

impl RunReport {
    /// Append an outcome.
    pub fn push(&mut self, outcome: CaseOutcome) {
        self.outcomes.push(outcome);
    }

    /// Number of outcomes with the given status.
    pub fn count(&self, status: CaseStatus) -> usize {
        self.outcomes.iter().filter(|o| o.status == status).count()
    }

    /// Whether any outcome is an overall failure.
    pub fn any_failures(&self) -> bool {
        self.outcomes.iter().any(|o| o.status.is_failure())
    }

    /// A one-line summary, e.g. `5 passed, 1 failed, 2 skipped`.
    pub fn summary(&self) -> String {
        let passed = self.count(CaseStatus::Pass) + self.count(CaseStatus::Xfail);
        let failed = self.count(CaseStatus::Fail) + self.count(CaseStatus::Xpass);
        let skipped = self.count(CaseStatus::Skip);
        format!("{passed} passed, {failed} failed, {skipped} skipped")
    }

    /// Render the results as CSV (`config,scenario,case,status,elapsed_s,returncode`).
    pub fn to_csv(&self) -> String {
        let mut out = String::from("config,scenario,case,status,elapsed_s,returncode\n");
        for o in &self.outcomes {
            out.push_str(&format!(
                "{},{},{},{},{:.4},{}\n",
                csv_escape(&o.config),
                csv_escape(&o.scenario),
                csv_escape(&o.case),
                o.status.label(),
                o.elapsed_s,
                o.returncode
            ));
        }
        out
    }
}

fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(case: &str, status: CaseStatus) -> CaseOutcome {
        CaseOutcome {
            case: case.into(),
            scenario: "rocjitsu".into(),
            config: "gfx1250".into(),
            status,
            elapsed_s: 0.5,
            returncode: 0,
            message: String::new(),
        }
    }

    #[test]
    fn summary_counts() {
        let mut r = RunReport::default();
        r.push(outcome("a", CaseStatus::Pass));
        r.push(outcome("b", CaseStatus::Fail));
        r.push(outcome("c", CaseStatus::Skip));
        r.push(outcome("d", CaseStatus::Xfail));
        assert_eq!(r.summary(), "2 passed, 1 failed, 1 skipped");
        assert!(r.any_failures());
    }

    #[test]
    fn csv_has_header_and_rows() {
        let mut r = RunReport::default();
        r.push(outcome("a", CaseStatus::Pass));
        let csv = r.to_csv();
        assert!(csv.starts_with("config,scenario,case,status,elapsed_s,returncode\n"));
        assert!(csv.contains("gfx1250,rocjitsu,a,pass"));
    }
}
