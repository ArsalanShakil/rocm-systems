//! Crate-wide error type for the corpus runner.

use std::path::PathBuf;

use thiserror::Error;

/// Result alias used throughout the corpus crate.
pub type Result<T> = std::result::Result<T, CorpusError>;

/// Everything that can go wrong while loading, generating, running, or
/// validating a corpus case.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// An I/O error tied to a specific path.
    #[error("io error on {path}: {source}")]
    Io {
        /// The path the operation was acting on.
        path: PathBuf,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A JSON document failed to parse.
    #[error("json error on {path}: {source}")]
    Json {
        /// The document path.
        path: PathBuf,
        /// The underlying serde error.
        #[source]
        source: serde_json::Error,
    },

    /// A TOML document failed to parse.
    #[error("toml error on {path}: {source}")]
    Toml {
        /// The document path.
        path: PathBuf,
        /// The underlying toml error.
        #[source]
        source: toml::de::Error,
    },

    /// A case or config document was structurally invalid.
    #[error("invalid case {path}: {message}")]
    Invalid {
        /// The offending document.
        path: PathBuf,
        /// What was wrong.
        message: String,
    },

    /// A CEL expression failed to compile or evaluate.
    #[error("cel error in {context}: {message}")]
    Cel {
        /// Where the expression came from (e.g. `inputs[0].formula`).
        context: String,
        /// The compile/evaluation error.
        message: String,
    },

    /// An external tool (iree-compile / iree-run-module / mirage) failed.
    #[error("{phase} failed (exit {code}): {tool}\n{log}")]
    Tool {
        /// `compile` or `run`.
        phase: String,
        /// The tool that was invoked.
        tool: String,
        /// The process exit code (127 = not found).
        code: i32,
        /// A short log excerpt (stdout/stderr).
        log: String,
    },

    /// Validation of an observed output failed.
    #[error("validation failed: {0}")]
    Validation(String),

    /// A required external tool was not found on `PATH`.
    #[error("required tool not found on PATH: {0}")]
    ToolMissing(String),

    /// A scenario could not be prepared (e.g. emulator not installed).
    /// These are surfaced as *skips*, not failures.
    #[error("scenario unavailable: {0}")]
    ScenarioUnavailable(String),

    /// A catch-all for anything else.
    #[error("{0}")]
    Other(String),
}

impl CorpusError {
    /// Build a generic error from a message.
    pub fn other(msg: impl Into<String>) -> Self {
        CorpusError::Other(msg.into())
    }

    /// Build an [`CorpusError::Invalid`] for a document path.
    pub fn invalid(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        CorpusError::Invalid {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Build a [`CorpusError::Cel`].
    pub fn cel(context: impl Into<String>, message: impl Into<String>) -> Self {
        CorpusError::Cel {
            context: context.into(),
            message: message.into(),
        }
    }
}
