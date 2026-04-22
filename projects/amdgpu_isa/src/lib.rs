//! AMD GPU ISA instruction definitions, encoder, and decoder.
//!
//! This crate provides:
//! - **Schema types** for parsing AMD GPU machine-readable ISA XML specifications
//! - **Generated instruction types** for each supported GPU architecture
//! - **Encoder/Decoder** for converting between binary and structured instructions
//! - **SP3 assembly display** via `Display` trait implementations
//!
//! # Architecture support
//!
//! Enable features for the architectures you need:
//! - `rdna1` through `rdna4` for RDNA architectures
//! - `cdna1` through `cdna4` for CDNA architectures

pub mod schema;

// ─── Core traits ───────────────────────────────────────────────────────────────

/// Trait implemented by all generated instruction enums.
pub trait Instruction: core::fmt::Display + core::fmt::Debug + Clone {
    /// The SP3 assembly mnemonic (e.g., `"v_add_f32"`).
    fn mnemonic(&self) -> &'static str;

    /// The encoding format name (e.g., `"ENC_VOP2"`).
    fn encoding_name(&self) -> &'static str;

    /// Whether this instruction is a branch.
    fn is_branch(&self) -> bool;

    /// Whether this instruction terminates the program.
    fn is_program_terminator(&self) -> bool;

    /// Encode this instruction to little-endian bytes.
    fn encode(&self) -> Vec<u8>;
}

/// Trait implemented by each ISA architecture.
pub trait Isa {
    /// The instruction enum type for this architecture.
    type Instruction: Instruction;

    /// Architecture name (e.g., `"AMD RDNA 4"`).
    fn name() -> &'static str;

    /// Decode an instruction from little-endian bytes.
    ///
    /// Returns the decoded instruction and the number of bytes consumed.
    fn decode(bytes: &[u8]) -> Result<(Self::Instruction, usize), DecodeError>;
}

// ─── Error types ───────────────────────────────────────────────────────────────

/// Errors that can occur during instruction decoding.
#[derive(Debug, Clone)]
pub enum DecodeError {
    /// Not enough bytes to decode an instruction.
    InsufficientBytes { needed: usize, available: usize },
    /// The instruction word doesn't match any known encoding.
    UnknownInstruction(u32),
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::InsufficientBytes { needed, available } => {
                write!(f, "insufficient bytes: need {needed}, have {available}")
            }
            DecodeError::UnknownInstruction(word) => {
                write!(f, "unknown instruction: {word:#010x}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

// ─── Operand formatting helpers ────────────────────────────────────────────────

/// Format a source operand value for SP3 assembly.
///
/// Decodes inline constants, SGPR references, VGPR references, and special registers.
pub fn format_src(value: u16) -> String {
    match value {
        0..=105 => format!("s{value}"),
        106 => "vcc_lo".to_string(),
        107 => "vcc_hi".to_string(),
        108 => "ttmp0".to_string(),
        109 => "ttmp1".to_string(),
        110 => "ttmp2".to_string(),
        111 => "ttmp3".to_string(),
        112 => "ttmp4".to_string(),
        113 => "ttmp5".to_string(),
        114 => "ttmp6".to_string(),
        115 => "ttmp7".to_string(),
        116 => "ttmp8".to_string(),
        117 => "ttmp9".to_string(),
        118 => "ttmp10".to_string(),
        119 => "ttmp11".to_string(),
        120 => "ttmp12".to_string(),
        121 => "ttmp13".to_string(),
        122 => "ttmp14".to_string(),
        123 => "ttmp15".to_string(),
        124 => "m0".to_string(),
        125 => "null".to_string(),
        126 => "exec_lo".to_string(),
        127 => "exec_hi".to_string(),
        128 => "0".to_string(),
        v @ 129..=192 => format!("{}", v as i32 - 128),
        v @ 193..=208 => format!("{}", -(v as i32 - 192)),
        240 => "0.5".to_string(),
        241 => "-0.5".to_string(),
        242 => "1.0".to_string(),
        243 => "-1.0".to_string(),
        244 => "2.0".to_string(),
        245 => "-2.0".to_string(),
        246 => "4.0".to_string(),
        247 => "-4.0".to_string(),
        255 => "lit".to_string(),
        v @ 256..=511 => format!("v{}", v - 256),
        v => format!("{v:#x}"),
    }
}

// ─── Generated ISA modules ────────────────────────────────────────────────────

#[cfg(feature = "rdna1")]
#[allow(unused_variables, unused_mut)]
pub mod rdna1 {
    include!(concat!(env!("OUT_DIR"), "/rdna1.rs"));
}

#[cfg(feature = "rdna2")]
#[allow(unused_variables, unused_mut)]
pub mod rdna2 {
    include!(concat!(env!("OUT_DIR"), "/rdna2.rs"));
}

#[cfg(feature = "rdna3")]
#[allow(unused_variables, unused_mut)]
pub mod rdna3 {
    include!(concat!(env!("OUT_DIR"), "/rdna3.rs"));
}

#[cfg(feature = "rdna3_5")]
#[allow(unused_variables, unused_mut)]
pub mod rdna3_5 {
    include!(concat!(env!("OUT_DIR"), "/rdna3_5.rs"));
}

#[cfg(feature = "rdna4")]
#[allow(unused_variables, unused_mut)]
pub mod rdna4 {
    include!(concat!(env!("OUT_DIR"), "/rdna4.rs"));
}

#[cfg(feature = "cdna1")]
#[allow(unused_variables, unused_mut)]
pub mod cdna1 {
    include!(concat!(env!("OUT_DIR"), "/cdna1.rs"));
}

#[cfg(feature = "cdna2")]
#[allow(unused_variables, unused_mut)]
pub mod cdna2 {
    include!(concat!(env!("OUT_DIR"), "/cdna2.rs"));
}

#[cfg(feature = "cdna3")]
#[allow(unused_variables, unused_mut)]
pub mod cdna3 {
    include!(concat!(env!("OUT_DIR"), "/cdna3.rs"));
}

#[cfg(feature = "cdna4")]
#[allow(unused_variables, unused_mut)]
pub mod cdna4 {
    include!(concat!(env!("OUT_DIR"), "/cdna4.rs"));
}