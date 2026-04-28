//! Schema types for parsing AMD GPU machine-readable ISA XML specifications.
//!
//! These types mirror the XML schema documented at:
//! <https://github.com/GPUOpen-Tools/isa_spec_manager/blob/main/documentation/spec_documentation.md>
//!
//! This module is self-contained so it can be shared between the library and build script.

#![allow(dead_code)]

/// Top-level specification containing document metadata and the ISA definition.
#[derive(Debug, Clone)]
pub struct Spec {
    pub document: Document,
    pub isa: IsaSpec,
}

/// Document metadata (copyright, version, release date).
#[derive(Debug, Clone)]
pub struct Document {
    pub copyright: String,
    pub sensitivity: String,
    pub license: Option<String>,
    pub release_date: String,
    pub schema_version: String,
}

/// The complete ISA specification for one GPU architecture.
#[derive(Debug, Clone)]
pub struct IsaSpec {
    pub architecture: Architecture,
    pub encodings: Vec<Encoding>,
    pub instructions: Vec<InstructionDef>,
    pub data_formats: Vec<DataFormat>,
    pub operand_types: Vec<OperandTypeDef>,
    pub functional_groups: Vec<FunctionalGroupDef>,
    pub functional_subgroups: Vec<FunctionalSubgroupDef>,
}

/// Architecture identification.
#[derive(Debug, Clone)]
pub struct Architecture {
    pub name: String,
    pub id: Option<u32>,
}

/// An encoding format (e.g., ENC_VOP2, ENC_SOP1).
#[derive(Debug, Clone)]
pub struct Encoding {
    pub order: i32,
    pub name: String,
    pub bit_count: u32,
    pub identifier_mask: u64,
    pub identifiers: Vec<u64>,
    pub conditions: Vec<EncodingConditionDef>,
    pub description: String,
    pub fields: Vec<MicrocodeField>,
}

/// A field in the encoding's microcode format bitmap.
#[derive(Debug, Clone)]
pub struct MicrocodeField {
    pub name: String,
    pub description: Option<String>,
    pub is_conditional: bool,
    pub ranges: Vec<BitRange>,
}

impl MicrocodeField {
    /// Total number of bits across all ranges (excluding padding).
    pub fn total_bits(&self) -> u32 {
        self.ranges.iter().map(|r| r.bit_count).sum()
    }
}

/// A contiguous bit range within an encoding field.
#[derive(Debug, Clone)]
pub struct BitRange {
    pub order: u32,
    pub bit_count: u32,
    pub bit_offset: u32,
    /// If present, padding bits are appended when reconstructing the value.
    pub padding: Option<Padding>,
}

/// Padding information for a bit range (e.g., SBASE in SMEM encodes bits [6:1]).
#[derive(Debug, Clone)]
pub struct Padding {
    pub bit_count: u32,
    pub value: u64,
}

/// A named encoding condition (e.g., "default", "has_lit").
#[derive(Debug, Clone)]
pub struct EncodingConditionDef {
    pub name: String,
}

/// Definition of a single ISA instruction.
#[derive(Debug, Clone)]
pub struct InstructionDef {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub flags: InstructionFlags,
    pub encodings: Vec<InstructionEncoding>,
    pub functional_group: Option<FunctionalGroup>,
}

/// Instruction behavioral flags.
#[derive(Debug, Clone, Default)]
pub struct InstructionFlags {
    pub is_branch: bool,
    pub is_conditional_branch: bool,
    pub is_indirect_branch: bool,
    pub is_program_terminator: bool,
    pub is_immediately_executed: bool,
}

/// One encoding variant of an instruction (encoding name + opcode + operands).
#[derive(Debug, Clone)]
pub struct InstructionEncoding {
    pub encoding_name: String,
    pub encoding_condition: String,
    pub opcode: u32,
    pub operands: Vec<OperandDef>,
}

/// A single operand of an instruction in a specific encoding.
#[derive(Debug, Clone)]
pub struct OperandDef {
    pub field_name: Option<String>,
    pub data_format_name: String,
    pub operand_type: String,
    pub operand_size: u32,
    pub is_input: bool,
    pub is_output: bool,
    pub is_implicit: bool,
    pub is_binary_microcode_required: bool,
    pub order: u32,
}

/// Data format definition (e.g., FMT_NUM_F32).
#[derive(Debug, Clone)]
pub struct DataFormat {
    pub name: String,
    pub description: String,
    pub data_type: String,
    pub bit_count: Option<u32>,
    pub component_count: u32,
}

/// Operand type definition (e.g., OPR_VGPR, OPR_SRC).
#[derive(Debug, Clone)]
pub struct OperandTypeDef {
    pub name: String,
    pub description: String,
    pub is_partitioned: bool,
    pub subtypes: Vec<String>,
    pub predefined_values: Vec<PredefinedValue>,
}

/// Maps an encoded integer value to an assembly name.
#[derive(Debug, Clone)]
pub struct PredefinedValue {
    pub name: String,
    pub description: Option<String>,
    pub value: u64,
}

/// Functional group classification for an instruction.
#[derive(Debug, Clone)]
pub struct FunctionalGroup {
    pub name: String,
    pub subgroups: Vec<String>,
}

/// Top-level functional group definition.
#[derive(Debug, Clone)]
pub struct FunctionalGroupDef {
    pub name: String,
    pub description: Option<String>,
}

/// Top-level functional subgroup definition.
#[derive(Debug, Clone)]
pub struct FunctionalSubgroupDef {
    pub name: String,
}

// ─── Parsing ───────────────────────────────────────────────────────────────────

/// Errors that can occur during XML parsing.
#[derive(Debug)]
pub enum ParseError {
    Xml(roxmltree::Error),
    MissingElement(&'static str),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Xml(e) => write!(f, "XML parse error: {e}"),
            ParseError::MissingElement(name) => write!(f, "missing required element: {name}"),
            ParseError::InvalidValue(msg) => write!(f, "invalid value: {msg}"),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<roxmltree::Error> for ParseError {
    fn from(e: roxmltree::Error) -> Self {
        ParseError::Xml(e)
    }
}

type Node<'a, 'b> = roxmltree::Node<'a, 'b>;

impl Spec {
    /// Parse an XML ISA specification string into structured types.
    pub fn parse(xml: &str) -> Result<Self, ParseError> {
        let doc = roxmltree::Document::parse(xml)?;
        let root = doc.root_element();
        expect_tag(&root, "Spec")?;

        let document = parse_document(&child_elem(&root, "Document")?)?;
        let isa = parse_isa(&child_elem(&root, "ISA")?)?;

        Ok(Spec { document, isa })
    }
}

// ─── Helper functions ──────────────────────────────────────────────────────────

fn expect_tag(node: &Node<'_, '_>, expected: &'static str) -> Result<(), ParseError> {
    if node.tag_name().name() != expected {
        return Err(ParseError::MissingElement(expected));
    }
    Ok(())
}

fn child_elem<'a, 'b>(
    parent: &Node<'a, 'b>,
    tag: &'static str,
) -> Result<Node<'a, 'b>, ParseError> {
    parent
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == tag)
        .ok_or(ParseError::MissingElement(tag))
}

fn opt_child_elem<'a, 'b>(parent: &Node<'a, 'b>, tag: &str) -> Option<Node<'a, 'b>> {
    parent
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == tag)
}

fn child_text(parent: &Node<'_, '_>, tag: &'static str) -> Result<String, ParseError> {
    let node = child_elem(parent, tag)?;
    Ok(node.text().unwrap_or_default().trim().to_string())
}

fn opt_child_text(parent: &Node<'_, '_>, tag: &str) -> Option<String> {
    opt_child_elem(parent, tag).and_then(|n| n.text().map(|t| t.trim().to_string()))
}

fn children_with_tag<'a, 'b>(parent: &Node<'a, 'b>, tag: &str) -> Vec<Node<'a, 'b>> {
    parent
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == tag)
        .collect()
}

fn parse_bool_flag(text: &str) -> bool {
    matches!(text.trim(), "TRUE" | "true" | "1")
}

fn parse_radix_u64(text: &str, radix_attr: Option<&str>) -> Result<u64, ParseError> {
    let text = text.trim();
    let radix = match radix_attr {
        Some("2") => 2,
        Some("16") => 16,
        Some("10") | None => 10,
        Some(other) => return Err(ParseError::InvalidValue(format!("unknown radix: {other}"))),
    };
    u64::from_str_radix(text, radix)
        .map_err(|e| ParseError::InvalidValue(format!("{e}: '{text}' radix={radix}")))
}

fn parse_radix_u32(text: &str, radix_attr: Option<&str>) -> Result<u32, ParseError> {
    let val = parse_radix_u64(text, radix_attr)?;
    Ok(val as u32)
}

// ─── Section parsers ───────────────────────────────────────────────────────────

fn parse_document(node: &Node<'_, '_>) -> Result<Document, ParseError> {
    Ok(Document {
        copyright: child_text(node, "Copyright")?,
        sensitivity: child_text(node, "Sensitivity")?,
        license: opt_child_text(node, "License"),
        release_date: child_text(node, "ReleaseDate")?,
        schema_version: child_text(node, "SchemaVersion")?,
    })
}

fn parse_isa(node: &Node<'_, '_>) -> Result<IsaSpec, ParseError> {
    let arch_node = child_elem(node, "Architecture")?;
    let architecture = Architecture {
        name: child_text(&arch_node, "ArchitectureName")?,
        id: opt_child_text(&arch_node, "ArchitectureId").and_then(|s| s.parse().ok()),
    };

    let encodings_node = child_elem(node, "Encodings")?;
    let encodings = children_with_tag(&encodings_node, "Encoding")
        .iter()
        .map(parse_encoding)
        .collect::<Result<Vec<_>, _>>()?;

    let instructions_node = child_elem(node, "Instructions")?;
    let instructions = children_with_tag(&instructions_node, "Instruction")
        .iter()
        .map(parse_instruction)
        .collect::<Result<Vec<_>, _>>()?;

    let data_formats_node = child_elem(node, "DataFormats")?;
    let data_formats = children_with_tag(&data_formats_node, "DataFormat")
        .iter()
        .map(parse_data_format)
        .collect::<Result<Vec<_>, _>>()?;

    let operand_types_node = child_elem(node, "OperandTypes")?;
    let operand_types = children_with_tag(&operand_types_node, "OperandType")
        .iter()
        .map(parse_operand_type)
        .collect::<Result<Vec<_>, _>>()?;

    let functional_groups = if let Some(fg_node) = opt_child_elem(node, "FunctionalGroups") {
        children_with_tag(&fg_node, "FunctionalGroup")
            .iter()
            .map(|n| {
                Ok(FunctionalGroupDef {
                    name: child_text(n, "Name")?,
                    description: opt_child_text(n, "Description"),
                })
            })
            .collect::<Result<Vec<_>, ParseError>>()?
    } else {
        vec![]
    };

    let functional_subgroups = if let Some(fsg_node) = opt_child_elem(node, "FunctionalSubgroups") {
        children_with_tag(&fsg_node, "FunctionalSubgroup")
            .iter()
            .map(|n| {
                Ok(FunctionalSubgroupDef {
                    name: child_text(n, "Name")?,
                })
            })
            .collect::<Result<Vec<_>, ParseError>>()?
    } else {
        vec![]
    };

    Ok(IsaSpec {
        architecture,
        encodings,
        instructions,
        data_formats,
        operand_types,
        functional_groups,
        functional_subgroups,
    })
}

fn parse_encoding(node: &Node<'_, '_>) -> Result<Encoding, ParseError> {
    let order: i32 = node
        .attribute("Order")
        .unwrap_or("0")
        .parse()
        .map_err(|e| ParseError::InvalidValue(format!("Order: {e}")))?;

    let name = child_text(node, "EncodingName")?;
    let bit_count: u32 = child_text(node, "BitCount")?
        .parse()
        .map_err(|e| ParseError::InvalidValue(format!("BitCount: {e}")))?;

    let mask_node = child_elem(node, "EncodingIdentifierMask")?;
    let identifier_mask = parse_radix_u64(
        mask_node.text().unwrap_or("0"),
        mask_node.attribute("Radix"),
    )?;

    let ids_node = child_elem(node, "EncodingIdentifiers")?;
    let identifiers = children_with_tag(&ids_node, "EncodingIdentifier")
        .iter()
        .map(|n| parse_radix_u64(n.text().unwrap_or("0"), n.attribute("Radix")))
        .collect::<Result<Vec<_>, _>>()?;

    let conditions = if let Some(conds_node) = opt_child_elem(node, "EncodingConditions") {
        children_with_tag(&conds_node, "EncodingCondition")
            .iter()
            .map(|n| {
                Ok(EncodingConditionDef {
                    name: child_text(n, "ConditionName")?,
                })
            })
            .collect::<Result<Vec<_>, ParseError>>()?
    } else {
        vec![]
    };

    let description = opt_child_text(node, "Description").unwrap_or_default();

    let fields = if let Some(mcf_node) = opt_child_elem(node, "MicrocodeFormat") {
        if let Some(bitmap_node) = opt_child_elem(&mcf_node, "BitMap") {
            children_with_tag(&bitmap_node, "Field")
                .iter()
                .map(parse_microcode_field)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    Ok(Encoding {
        order,
        name,
        bit_count,
        identifier_mask,
        identifiers,
        conditions,
        description,
        fields,
    })
}

fn parse_microcode_field(node: &Node<'_, '_>) -> Result<MicrocodeField, ParseError> {
    let is_conditional = node
        .attribute("IsConditional")
        .map(parse_bool_flag)
        .unwrap_or(false);

    let name = child_text(node, "FieldName")?;
    let description = opt_child_text(node, "Description");

    let layout_node = child_elem(node, "BitLayout")?;
    let ranges = children_with_tag(&layout_node, "Range")
        .iter()
        .map(parse_bit_range)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(MicrocodeField {
        name,
        description,
        is_conditional,
        ranges,
    })
}

fn parse_bit_range(node: &Node<'_, '_>) -> Result<BitRange, ParseError> {
    let order: u32 = node
        .attribute("Order")
        .unwrap_or("0")
        .parse()
        .map_err(|e| ParseError::InvalidValue(format!("Range Order: {e}")))?;

    let bit_count: u32 = child_text(node, "BitCount")?
        .parse()
        .map_err(|e| ParseError::InvalidValue(format!("Range BitCount: {e}")))?;

    let bit_offset: u32 = child_text(node, "BitOffset")?
        .parse()
        .map_err(|e| ParseError::InvalidValue(format!("Range BitOffset: {e}")))?;

    let padding = if let Some(pad_node) = opt_child_elem(node, "Padding") {
        let pad_bits: u32 = child_text(&pad_node, "BitCount")?
            .parse()
            .map_err(|e| ParseError::InvalidValue(format!("Padding BitCount: {e}")))?;
        let val_node = child_elem(&pad_node, "Value")?;
        let pad_val = parse_radix_u64(val_node.text().unwrap_or("0"), val_node.attribute("Radix"))?;
        Some(Padding {
            bit_count: pad_bits,
            value: pad_val,
        })
    } else {
        None
    };

    Ok(BitRange {
        order,
        bit_count,
        bit_offset,
        padding,
    })
}

fn parse_instruction(node: &Node<'_, '_>) -> Result<InstructionDef, ParseError> {
    let flags_node = child_elem(node, "InstructionFlags")?;
    let flags = InstructionFlags {
        is_branch: opt_child_text(&flags_node, "IsBranch")
            .map(|s| parse_bool_flag(&s))
            .unwrap_or(false),
        is_conditional_branch: opt_child_text(&flags_node, "IsConditionalBranch")
            .map(|s| parse_bool_flag(&s))
            .unwrap_or(false),
        is_indirect_branch: opt_child_text(&flags_node, "IsIndirectBranch")
            .map(|s| parse_bool_flag(&s))
            .unwrap_or(false),
        is_program_terminator: opt_child_text(&flags_node, "IsProgramTerminator")
            .map(|s| parse_bool_flag(&s))
            .unwrap_or(false),
        is_immediately_executed: opt_child_text(&flags_node, "IsImmediatelyExecuted")
            .map(|s| parse_bool_flag(&s))
            .unwrap_or(false),
    };

    let name = child_text(node, "InstructionName")?;
    let description = opt_child_text(node, "Description").unwrap_or_default();

    let aliases = if let Some(aliases_node) = opt_child_elem(node, "AliasedInstructionNames") {
        children_with_tag(&aliases_node, "InstructionName")
            .iter()
            .filter_map(|n| n.text().map(|t| t.trim().to_string()))
            .collect()
    } else {
        vec![]
    };

    let enc_node = child_elem(node, "InstructionEncodings")?;
    let encodings = children_with_tag(&enc_node, "InstructionEncoding")
        .iter()
        .map(parse_instruction_encoding)
        .collect::<Result<Vec<_>, _>>()?;

    let functional_group: Option<FunctionalGroup> = opt_child_elem(node, "FunctionalGroup")
        .map(|fg_node| -> Result<FunctionalGroup, ParseError> {
            let fg_name = child_text(&fg_node, "Name")?;
            let subgroups = if let Some(sgs_node) = opt_child_elem(&fg_node, "FunctionalSubgroups")
            {
                children_with_tag(&sgs_node, "Subgroup")
                    .iter()
                    .filter_map(|n| n.text().map(|t| t.trim().to_string()))
                    .collect()
            } else {
                vec![]
            };
            Ok(FunctionalGroup {
                name: fg_name,
                subgroups,
            })
        })
        .transpose()?;

    Ok(InstructionDef {
        name,
        aliases,
        description,
        flags,
        encodings,
        functional_group,
    })
}

fn parse_instruction_encoding(node: &Node<'_, '_>) -> Result<InstructionEncoding, ParseError> {
    let encoding_name = child_text(node, "EncodingName")?;
    let encoding_condition = opt_child_text(node, "EncodingCondition").unwrap_or_default();

    let opcode_node = child_elem(node, "Opcode")?;
    let opcode = parse_radix_u32(
        opcode_node.text().unwrap_or("0"),
        opcode_node.attribute("Radix"),
    )?;

    let operands = if let Some(ops_node) = opt_child_elem(node, "Operands") {
        children_with_tag(&ops_node, "Operand")
            .iter()
            .map(parse_operand)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![]
    };

    Ok(InstructionEncoding {
        encoding_name,
        encoding_condition,
        opcode,
        operands,
    })
}

fn parse_operand(node: &Node<'_, '_>) -> Result<OperandDef, ParseError> {
    let is_input = node
        .attribute("Input")
        .map(parse_bool_flag)
        .unwrap_or(false);
    let is_output = node
        .attribute("Output")
        .map(parse_bool_flag)
        .unwrap_or(false);
    let is_implicit = node
        .attribute("IsImplicit")
        .map(parse_bool_flag)
        .unwrap_or(false);
    let is_binary_microcode_required = node
        .attribute("IsBinaryMicrocodeRequired")
        .map(parse_bool_flag)
        .unwrap_or(false);
    let order: u32 = node
        .attribute("Order")
        .unwrap_or("0")
        .parse()
        .map_err(|e| ParseError::InvalidValue(format!("Operand Order: {e}")))?;

    let field_name = opt_child_text(node, "FieldName");
    let data_format_name = child_text(node, "DataFormatName")?;
    let operand_type = child_text(node, "OperandType")?;
    let operand_size: u32 = child_text(node, "OperandSize")?
        .parse()
        .map_err(|e| ParseError::InvalidValue(format!("OperandSize: {e}")))?;

    Ok(OperandDef {
        field_name,
        data_format_name,
        operand_type,
        operand_size,
        is_input,
        is_output,
        is_implicit,
        is_binary_microcode_required,
        order,
    })
}

fn parse_data_format(node: &Node<'_, '_>) -> Result<DataFormat, ParseError> {
    Ok(DataFormat {
        name: child_text(node, "DataFormatName")?,
        description: opt_child_text(node, "Description").unwrap_or_default(),
        data_type: opt_child_text(node, "DataType").unwrap_or_default(),
        bit_count: opt_child_text(node, "BitCount").and_then(|s| s.parse().ok()),
        component_count: opt_child_text(node, "ComponentCount")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
    })
}

fn parse_operand_type(node: &Node<'_, '_>) -> Result<OperandTypeDef, ParseError> {
    let is_partitioned = node
        .attribute("IsPartitioned")
        .map(parse_bool_flag)
        .unwrap_or(false);

    let name = child_text(node, "OperandTypeName")?;
    let description = opt_child_text(node, "Description").unwrap_or_default();

    let subtypes = if let Some(st_node) = opt_child_elem(node, "Subtypes") {
        children_with_tag(&st_node, "SubtypeName")
            .iter()
            .filter_map(|n| n.text().map(|t| t.trim().to_string()))
            .collect()
    } else {
        vec![]
    };

    let predefined_values = if let Some(pv_node) = opt_child_elem(node, "OperandPredefinedValues") {
        children_with_tag(&pv_node, "PredefinedValue")
            .iter()
            .map(|n| {
                Ok(PredefinedValue {
                    name: child_text(n, "Name")?,
                    description: opt_child_text(n, "Description"),
                    value: child_text(n, "Value")?
                        .parse()
                        .map_err(|e| ParseError::InvalidValue(format!("PV Value: {e}")))?,
                })
            })
            .collect::<Result<Vec<_>, ParseError>>()?
    } else {
        vec![]
    };

    Ok(OperandTypeDef {
        name,
        description,
        is_partitioned,
        subtypes,
        predefined_values,
    })
}
