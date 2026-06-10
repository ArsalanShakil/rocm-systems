//! Scenarios: named ways to run a kernel under a mirage emulator.
//!
//! A scenario maps to a mirage *profile* pinning an emulator backend and a
//! GPU agent. The runner ensures the profile exists, then wraps each
//! `iree-run-module` invocation in `mirage run --profile <profile> --`.

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDef, EmulatorKind, ExecMode};
use mirage_core::profile::ProfileDef;
use mirage_core::topology::TopologyDef;

use crate::error::{CorpusError, Result};
use crate::model::Scenario;

/// The emulator backends a scenario can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioKind {
    /// rocjitsu CPU functional model.
    Rocjitsu,
    /// HotSwap ISA rewriter on a real GPU.
    Hotswap,
    /// Native pass-through (`noop`) — run on the host as-is.
    Native,
}

impl ScenarioKind {
    /// Parse a scenario name (`rocjitsu`, `hotswap`, `native`).
    pub fn parse(name: &str) -> Result<ScenarioKind> {
        match name {
            "rocjitsu" => Ok(ScenarioKind::Rocjitsu),
            "hotswap" => Ok(ScenarioKind::Hotswap),
            "native" | "noop" => Ok(ScenarioKind::Native),
            other => Err(CorpusError::other(format!(
                "unknown scenario {other:?} (expected rocjitsu, hotswap, or native)"
            ))),
        }
    }

    /// The mirage emulator backend this scenario uses.
    pub fn emulator(&self) -> EmulatorKind {
        match self {
            ScenarioKind::Rocjitsu => EmulatorKind::Rocjitsu,
            ScenarioKind::Hotswap => EmulatorKind::Hotswap,
            ScenarioKind::Native => EmulatorKind::Noop,
        }
    }

    /// The default builtin GPU agent the profile pins. HotSwap only
    /// supports MI450X today; the others default to MI300X.
    pub fn default_agent(&self) -> &'static str {
        match self {
            ScenarioKind::Hotswap => "MI450X",
            _ => "MI300X",
        }
    }

    /// Stable scenario name.
    pub fn name(&self) -> &'static str {
        match self {
            ScenarioKind::Rocjitsu => "rocjitsu",
            ScenarioKind::Hotswap => "hotswap",
            ScenarioKind::Native => "native",
        }
    }

    /// One-line description.
    pub fn description(&self) -> &'static str {
        match self {
            ScenarioKind::Rocjitsu => "rocjitsu CPU functional model (no GPU required)",
            ScenarioKind::Hotswap => "HotSwap ISA rewriter on a real AMD GPU",
            ScenarioKind::Native => "native pass-through on the host runtime",
        }
    }

    /// The mirage profile name this scenario creates / reuses.
    pub fn profile_name(&self) -> String {
        format!("corpus-{}", self.name())
    }

    /// Build the [`Scenario`] descriptor.
    pub fn scenario(&self) -> Scenario {
        Scenario {
            name: self.name().to_string(),
            description: self.description().to_string(),
            emulator: self.emulator().as_str().to_string(),
            agent: self.default_agent().to_string(),
            profile: self.profile_name(),
        }
    }

    /// Build the mirage [`ProfileDef`] for this scenario.
    pub fn profile_def(&self) -> ProfileDef {
        let topology = TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref(self.default_agent().to_string()),
        };
        ProfileDef {
            name: self.profile_name(),
            description: Some(format!("rocjitsu corpus scenario: {}", self.description())),
            emulator: EmulatorDef {
                emulator: self.emulator(),
                plugins: Default::default(),
                exec_mode: ExecMode::Functional,
                options: Default::default(),
                topology: MaybeRef::Owned(topology),
            },
            containerize: None,
        }
    }

    /// The profile as a JSON document (for `mirage profile import -`).
    pub fn profile_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.profile_def())
            .map_err(|e| CorpusError::other(format!("serialize profile: {e}")))
    }
}

/// The default set of built-in scenarios the UI/CLI iterate over.
pub fn builtin_scenarios() -> Vec<Scenario> {
    [
        ScenarioKind::Rocjitsu,
        ScenarioKind::Hotswap,
        ScenarioKind::Native,
    ]
    .iter()
    .map(|k| k.scenario())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_emulator() {
        assert_eq!(ScenarioKind::parse("rocjitsu").unwrap(), ScenarioKind::Rocjitsu);
        assert_eq!(ScenarioKind::parse("native").unwrap().emulator(), EmulatorKind::Noop);
        assert!(ScenarioKind::parse("bogus").is_err());
    }

    #[test]
    fn profile_json_roundtrips() {
        let json = ScenarioKind::Rocjitsu.profile_json().unwrap();
        let def: ProfileDef = serde_json::from_str(&json).unwrap();
        assert_eq!(def.name, "corpus-rocjitsu");
        assert_eq!(def.emulator.emulator, EmulatorKind::Rocjitsu);
    }

    #[test]
    fn builtin_set_has_three() {
        let s = builtin_scenarios();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].name, "rocjitsu");
    }
}
