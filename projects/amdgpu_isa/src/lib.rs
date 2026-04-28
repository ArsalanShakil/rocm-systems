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

// ─── Encoding format enum ──────────────────────────────────────────────────────

/// All known GPU instruction encoding formats across RDNA and CDNA architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingFormat {
    EncDs,
    EncExp,
    EncFlat,
    EncFlatGlbl,
    EncFlatGlobal,
    EncFlatScratch,
    EncLdsdir,
    EncMimg,
    EncMtbuf,
    EncMubuf,
    EncSmem,
    EncSop1,
    EncSop2,
    EncSopc,
    EncSopk,
    EncSopp,
    EncVbuffer,
    EncVds,
    EncVdsdir,
    EncVexport,
    EncVflat,
    EncVglobal,
    EncVimage,
    EncVinterp,
    EncVintrp,
    EncVop1,
    EncVop2,
    EncVop3,
    EncVop3p,
    EncVop3px2,
    EncVopc,
    EncVsample,
    EncVscratch,
    MimgNsa1,
    MimgNsa2,
    MimgNsa3,
    Sop1InstLiteral,
    Sop2InstLiteral,
    SopcInstLiteral,
    SopkInstLiteral,
    Vop1InstLiteral,
    Vop1VopDpp,
    Vop1VopDpp16,
    Vop1VopDpp8,
    Vop1VopSdwa,
    Vop2InstLiteral,
    Vop2VopDpp,
    Vop2VopDpp16,
    Vop2VopDpp8,
    Vop2VopSdwa,
    Vop2VopSdwaSdstEnc,
    Vop3InstLiteral,
    Vop3SdstEnc,
    Vop3SdstEncInstLiteral,
    Vop3SdstEncVopDpp16,
    Vop3SdstEncVopDpp8,
    Vop3VopDpp16,
    Vop3VopDpp8,
    Vop3pInstLiteral,
    Vop3pMfma,
    Vop3pVopDpp16,
    Vop3pVopDpp8,
    VopcInstLiteral,
    VopcVopDpp16,
    VopcVopDpp8,
    VopcVopSdwaSdstEnc,
    Vopdxy,
    VopdxyInstLiteral,
}

impl EncodingFormat {
    /// Returns the canonical encoding name string (e.g., `"ENC_VOP2"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EncDs => "ENC_DS",
            Self::EncExp => "ENC_EXP",
            Self::EncFlat => "ENC_FLAT",
            Self::EncFlatGlbl => "ENC_FLAT_GLBL",
            Self::EncFlatGlobal => "ENC_FLAT_GLOBAL",
            Self::EncFlatScratch => "ENC_FLAT_SCRATCH",
            Self::EncLdsdir => "ENC_LDSDIR",
            Self::EncMimg => "ENC_MIMG",
            Self::EncMtbuf => "ENC_MTBUF",
            Self::EncMubuf => "ENC_MUBUF",
            Self::EncSmem => "ENC_SMEM",
            Self::EncSop1 => "ENC_SOP1",
            Self::EncSop2 => "ENC_SOP2",
            Self::EncSopc => "ENC_SOPC",
            Self::EncSopk => "ENC_SOPK",
            Self::EncSopp => "ENC_SOPP",
            Self::EncVbuffer => "ENC_VBUFFER",
            Self::EncVds => "ENC_VDS",
            Self::EncVdsdir => "ENC_VDSDIR",
            Self::EncVexport => "ENC_VEXPORT",
            Self::EncVflat => "ENC_VFLAT",
            Self::EncVglobal => "ENC_VGLOBAL",
            Self::EncVimage => "ENC_VIMAGE",
            Self::EncVinterp => "ENC_VINTERP",
            Self::EncVintrp => "ENC_VINTRP",
            Self::EncVop1 => "ENC_VOP1",
            Self::EncVop2 => "ENC_VOP2",
            Self::EncVop3 => "ENC_VOP3",
            Self::EncVop3p => "ENC_VOP3P",
            Self::EncVop3px2 => "ENC_VOP3PX2",
            Self::EncVopc => "ENC_VOPC",
            Self::EncVsample => "ENC_VSAMPLE",
            Self::EncVscratch => "ENC_VSCRATCH",
            Self::MimgNsa1 => "MIMG_NSA1",
            Self::MimgNsa2 => "MIMG_NSA2",
            Self::MimgNsa3 => "MIMG_NSA3",
            Self::Sop1InstLiteral => "SOP1_INST_LITERAL",
            Self::Sop2InstLiteral => "SOP2_INST_LITERAL",
            Self::SopcInstLiteral => "SOPC_INST_LITERAL",
            Self::SopkInstLiteral => "SOPK_INST_LITERAL",
            Self::Vop1InstLiteral => "VOP1_INST_LITERAL",
            Self::Vop1VopDpp => "VOP1_VOP_DPP",
            Self::Vop1VopDpp16 => "VOP1_VOP_DPP16",
            Self::Vop1VopDpp8 => "VOP1_VOP_DPP8",
            Self::Vop1VopSdwa => "VOP1_VOP_SDWA",
            Self::Vop2InstLiteral => "VOP2_INST_LITERAL",
            Self::Vop2VopDpp => "VOP2_VOP_DPP",
            Self::Vop2VopDpp16 => "VOP2_VOP_DPP16",
            Self::Vop2VopDpp8 => "VOP2_VOP_DPP8",
            Self::Vop2VopSdwa => "VOP2_VOP_SDWA",
            Self::Vop2VopSdwaSdstEnc => "VOP2_VOP_SDWA_SDST_ENC",
            Self::Vop3InstLiteral => "VOP3_INST_LITERAL",
            Self::Vop3SdstEnc => "VOP3_SDST_ENC",
            Self::Vop3SdstEncInstLiteral => "VOP3_SDST_ENC_INST_LITERAL",
            Self::Vop3SdstEncVopDpp16 => "VOP3_SDST_ENC_VOP_DPP16",
            Self::Vop3SdstEncVopDpp8 => "VOP3_SDST_ENC_VOP_DPP8",
            Self::Vop3VopDpp16 => "VOP3_VOP_DPP16",
            Self::Vop3VopDpp8 => "VOP3_VOP_DPP8",
            Self::Vop3pInstLiteral => "VOP3P_INST_LITERAL",
            Self::Vop3pMfma => "VOP3P_MFMA",
            Self::Vop3pVopDpp16 => "VOP3P_VOP_DPP16",
            Self::Vop3pVopDpp8 => "VOP3P_VOP_DPP8",
            Self::VopcInstLiteral => "VOPC_INST_LITERAL",
            Self::VopcVopDpp16 => "VOPC_VOP_DPP16",
            Self::VopcVopDpp8 => "VOPC_VOP_DPP8",
            Self::VopcVopSdwaSdstEnc => "VOPC_VOP_SDWA_SDST_ENC",
            Self::Vopdxy => "VOPDXY",
            Self::VopdxyInstLiteral => "VOPDXY_INST_LITERAL",
        }
    }
}

impl core::fmt::Display for EncodingFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── Core traits ───────────────────────────────────────────────────────────────

/// Trait implemented by all generated instruction enums.
pub trait Instruction: core::fmt::Display + core::fmt::Debug + Clone {
    /// The SP3 assembly mnemonic (e.g., `"v_add_f32"`).
    fn mnemonic(&self) -> &'static str;

    /// The encoding format for this instruction.
    fn encoding_format(&self) -> EncodingFormat;

    /// Whether this instruction is a branch.
    fn is_branch(&self) -> bool;

    /// Whether this instruction terminates the program.
    fn is_program_terminator(&self) -> bool;

    /// Encode this instruction into `writer` as little-endian bytes.
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()>;
}

/// Trait implemented by each ISA architecture.
pub trait Isa {
    /// The instruction enum type for this architecture.
    type Instruction: Instruction;

    /// Architecture name (e.g., `"AMD RDNA 4"`).
    fn name() -> &'static str;

    /// Decode an instruction from `reader` (little-endian bytes).
    fn decode<R: std::io::Read>(reader: &mut R) -> Result<Self::Instruction, DecodeError>;
}

// ─── Error types ───────────────────────────────────────────────────────────────

/// Errors that can occur during instruction decoding.
#[derive(Debug)]
pub enum DecodeError {
    /// An I/O error occurred while reading from the input. Wraps the underlying
    /// `std::io::Error`. A `kind() == UnexpectedEof` indicates insufficient bytes.
    Io(std::io::Error),
    /// The instruction word doesn't match any known encoding.
    UnknownInstruction(u32),
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::Io(e) => write!(f, "io error: {e}"),
            DecodeError::UnknownInstruction(word) => {
                write!(f, "unknown instruction: {word:#010x}")
            }
        }
    }
}

impl From<std::io::Error> for DecodeError {
    fn from(e: std::io::Error) -> Self {
        DecodeError::Io(e)
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::Io(e) => Some(e),
            DecodeError::UnknownInstruction(_) => None,
        }
    }
}

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
