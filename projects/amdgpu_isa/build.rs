//! Build script that parses AMD GPU ISA XML specs and generates
//! Rust instruction types, encoders, decoders, and Display impls.

#[path = "src/schema.rs"]
mod schema;

use schema::*;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::path::Path;
use std::{env, fs};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let data_dir = Path::new(&manifest_dir).join("data");

    let isa_files: &[(&str, &str, &str)] = &[
        ("rdna1", "amdgpu_isa_rdna1.xml", "RDNA1"),
        ("rdna2", "amdgpu_isa_rdna2.xml", "RDNA2"),
        ("rdna3", "amdgpu_isa_rdna3.xml", "RDNA3"),
        ("rdna3_5", "amdgpu_isa_rdna3_5.xml", "RDNA3_5"),
        ("rdna4", "amdgpu_isa_rdna4.xml", "RDNA4"),
        ("cdna1", "amdgpu_isa_cdna1.xml", "CDNA1"),
        ("cdna2", "amdgpu_isa_cdna2.xml", "CDNA2"),
        ("cdna3", "amdgpu_isa_cdna3.xml", "CDNA3"),
        ("cdna4", "amdgpu_isa_cdna4.xml", "CDNA4"),
    ];

    for (feature, filename, isa_name) in isa_files {
        let xml_path = data_dir.join(filename);
        println!("cargo:rerun-if-changed={}", xml_path.display());

        if env::var(format!(
            "CARGO_FEATURE_{}",
            feature.to_uppercase()
        ))
        .is_ok()
        {
            if !xml_path.exists() {
                println!(
                    "cargo:warning=ISA XML file not found: {}",
                    xml_path.display()
                );
                continue;
            }

            let xml = fs::read_to_string(&xml_path).unwrap_or_else(|e| {
                panic!("Failed to read {}: {}", xml_path.display(), e);
            });

            let spec = Spec::parse(&xml).unwrap_or_else(|e| {
                panic!("Failed to parse {}: {}", filename, e);
            });

            let code = generate_isa_module(&spec, isa_name);

            let out_path = Path::new(&out_dir).join(format!("{feature}.rs"));
            let mut file = fs::File::create(&out_path).unwrap();
            file.write_all(code.as_bytes()).unwrap();
        }
    }
}

// ─── Code Generation ───────────────────────────────────────────────────────────

fn generate_isa_module(spec: &Spec, isa_name: &str) -> String {
    let mut out = String::with_capacity(1024 * 256);

    let encoding_map: HashMap<&str, &Encoding> = spec
        .isa
        .encodings
        .iter()
        .map(|e| (e.name.as_str(), e))
        .collect();

    // Build (encoding_name, opcode) → instruction index mapping for decode
    let mut enc_op_map: HashMap<(&str, u32), Vec<(usize, &str)>> = HashMap::new();
    for (idx, inst) in spec.isa.instructions.iter().enumerate() {
        for enc in &inst.encodings {
            enc_op_map
                .entry((enc.encoding_name.as_str(), enc.opcode))
                .or_default()
                .push((idx, enc.encoding_condition.as_str()));
        }
    }

    // Generate per-instruction encoding enums
    for inst in &spec.isa.instructions {
        generate_instruction_enum(&mut out, inst, &encoding_map);
    }

    // Generate main instruction enum
    generate_main_enum(&mut out, &spec.isa.instructions, isa_name);

    // Generate Display for main enum
    generate_main_display(&mut out, &spec.isa.instructions, isa_name);

    // Generate Instruction trait impl
    generate_instruction_trait_impl(&mut out, &spec.isa.instructions, isa_name);

    // Generate Isa struct + impl
    generate_isa_struct(
        &mut out,
        &spec.isa,
        isa_name,
        &encoding_map,
        &enc_op_map,
    );

    // Generate visitor trait
    generate_visitor_trait(&mut out, &spec.isa.instructions, isa_name);

    out
}

// ─── Per-instruction encoding enum ─────────────────────────────────────────────

fn generate_instruction_enum(
    out: &mut String,
    inst: &InstructionDef,
    encoding_map: &HashMap<&str, &Encoding>,
) {
    let type_name = to_pascal_case(&inst.name);

    // Doc comment
    let _ = writeln!(out, "/// `{}`: {}", inst.name, escape_doc(&inst.description));
    let _ = writeln!(out, "#[derive(Debug, Clone, PartialEq)]");
    let _ = writeln!(out, "pub enum {type_name} {{");

    let mut used_variants: HashMap<String, usize> = HashMap::new();
    for enc in &inst.encodings {
        let variant_name = unique_variant_name(
            &enc.encoding_name,
            &enc.encoding_condition,
            &mut used_variants,
        );
        let _ = writeln!(out, "    /// Encoding: `{}`, condition: `{}`", enc.encoding_name, enc.encoding_condition);

        // Collect non-implicit operands that have field names (i.e. present in binary)
        let fields = collect_fields(enc, encoding_map);

        if fields.is_empty() {
            let _ = writeln!(out, "    {variant_name},");
        } else {
            let _ = writeln!(out, "    {variant_name} {{");
            for (field_name, rust_type) in &fields {
                let _ = writeln!(out, "        {field_name}: {rust_type},");
            }
            let _ = writeln!(out, "    }},");
        }
    }

    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    // Display impl for this instruction's encoding enum
    generate_instruction_display(out, inst, encoding_map);

    // Encode impl
    generate_instruction_encode(out, inst, encoding_map);
}

fn generate_instruction_display(
    out: &mut String,
    inst: &InstructionDef,
    encoding_map: &HashMap<&str, &Encoding>,
) {
    let type_name = to_pascal_case(&inst.name);
    let mnemonic = inst.name.to_lowercase();

    let _ = writeln!(out, "impl core::fmt::Display for {type_name} {{");
    let _ = writeln!(
        out,
        "    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{"
    );
    let _ = writeln!(out, "        match self {{");

    let mut used_variants: HashMap<String, usize> = HashMap::new();
    for enc in &inst.encodings {
        let variant_name = unique_variant_name(
            &enc.encoding_name,
            &enc.encoding_condition,
            &mut used_variants,
        );
        let fields = collect_fields(enc, encoding_map);

        // Collect visible operands for assembly (non-implicit, in order)
        let mut visible_ops: Vec<&OperandDef> = enc
            .operands
            .iter()
            .filter(|op| !op.is_implicit && op.field_name.is_some())
            .collect();
        visible_ops.sort_by_key(|op| op.order);

        if fields.is_empty() {
            let _ = writeln!(
                out,
                "            Self::{variant_name} => write!(f, \"{mnemonic}\"),"
            );
        } else {
            // Build format string and args using positional args only
            let mut fmt_str = mnemonic.clone();
            let mut fmt_args: Vec<String> = Vec::new();
            let mut used_fields: std::collections::HashSet<String> = std::collections::HashSet::new();

            for (i, op) in visible_ops.iter().enumerate() {
                let fname = op.field_name.as_ref().unwrap();
                let rust_name = to_snake_case(fname);
                let actual_name = fields
                    .iter()
                    .find(|(n, _)| *n == rust_name || n.starts_with(&format!("{rust_name}_")))
                    .map(|(n, _)| n.as_str())
                    .unwrap_or(rust_name.as_str());

                used_fields.insert(actual_name.to_string());

                if i == 0 {
                    fmt_str.push(' ');
                } else {
                    fmt_str.push_str(", ");
                }

                match op.operand_type.as_str() {
                    "OPR_VGPR" => {
                        fmt_str.push_str(&format!("v{{{}}}", fmt_args.len()));
                        fmt_args.push(actual_name.to_string());
                    }
                    "OPR_SREG" | "OPR_SGPR" => {
                        fmt_str.push_str(&format!("s{{{}}}", fmt_args.len()));
                        fmt_args.push(actual_name.to_string());
                    }
                    "OPR_SRC" => {
                        fmt_str.push_str(&format!("{{{}}}", fmt_args.len()));
                        fmt_args.push(format!("crate::format_src(*{actual_name} as u16)"));
                    }
                    _ => {
                        let pos = fmt_args.len();
                        fmt_str.push('{');
                        fmt_str.push_str(&pos.to_string());
                        fmt_str.push_str(":#x}");
                        fmt_args.push(actual_name.to_string());
                    }
                }
            }

            // Prefix unused field bindings with _
            let field_bindings: Vec<String> = fields
                .iter()
                .map(|(name, _)| {
                    if used_fields.contains(name.as_str()) {
                        name.clone()
                    } else {
                        format!("{name}: _")
                    }
                })
                .collect();
            let binding_str = field_bindings.join(", ");
            let _ = writeln!(
                out,
                "            Self::{variant_name} {{ {binding_str} }} => {{"
            );

            let _ = write!(out, "                write!(f, \"{fmt_str}\"");
            for arg in &fmt_args {
                let _ = write!(out, ", {arg}");
            }
            let _ = writeln!(out, ")");
            let _ = writeln!(out, "            }}");
        }
    }

    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

fn generate_instruction_encode(
    out: &mut String,
    inst: &InstructionDef,
    encoding_map: &HashMap<&str, &Encoding>,
) {
    let type_name = to_pascal_case(&inst.name);

    let _ = writeln!(out, "impl {type_name} {{");
    let _ = writeln!(out, "    /// Encode this instruction to bytes (little-endian).");
    let _ = writeln!(out, "    pub fn encode(&self) -> Vec<u8> {{");
    let _ = writeln!(out, "        match self {{");

    let mut used_variants: HashMap<String, usize> = HashMap::new();
    for enc in &inst.encodings {
        let variant_name = unique_variant_name(
            &enc.encoding_name,
            &enc.encoding_condition,
            &mut used_variants,
        );
        let fields = collect_fields(enc, encoding_map);
        let enc_def = encoding_map.get(enc.encoding_name.as_str());

        if fields.is_empty() {
            let _ = writeln!(out, "            Self::{variant_name} => {{");
        } else {
            let binding_str = fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "            Self::{variant_name} {{ {binding_str} }} => {{"
            );
        }

        if let Some(enc_def) = enc_def {
            let bit_count = enc_def.bit_count;
            if bit_count <= 32 {
                let _ = writeln!(out, "                let mut word: u32 = 0;");
                if let Some(enc_field) = enc_def.fields.iter().find(|f| f.name == "ENCODING") {
                    let enc_value = derive_encoding_prefix(enc_def);
                    for range in &enc_field.ranges {
                        let mask = (1u64 << range.bit_count) - 1;
                        let _ = writeln!(
                            out,
                            "                word |= ({enc_value}u32 & {mask:#x}u32) << {};",
                            range.bit_offset
                        );
                    }
                }
                if let Some(op_field) = enc_def.fields.iter().find(|f| f.name == "OP") {
                    for range in &op_field.ranges {
                        let mask = (1u64 << range.bit_count) - 1;
                        let _ = writeln!(
                            out,
                            "                word |= ({}u32 & {mask:#x}u32) << {};",
                            enc.opcode, range.bit_offset
                        );
                    }
                }
                for (field_name, _rust_type) in &fields {
                    let orig_name = field_name.to_uppercase();
                    if let Some(mc_field) = enc_def.fields.iter().find(|f| {
                        f.name == orig_name || to_snake_case(&f.name) == *field_name
                    }) {
                        for range in &mc_field.ranges {
                            let mask = (1u64 << range.bit_count) - 1;
                            let _ = writeln!(
                                out,
                                "                word |= (*{field_name} as u32 & {mask:#x}u32) << {};",
                                range.bit_offset
                            );
                        }
                    }
                }
                let _ = writeln!(out, "                word.to_le_bytes().to_vec()");
            } else {
                let byte_count = bit_count / 8;
                let _ = writeln!(out, "                let mut word: u128 = 0;");
                if let Some(enc_field) = enc_def.fields.iter().find(|f| f.name == "ENCODING") {
                    let enc_value = derive_encoding_prefix(enc_def);
                    for range in &enc_field.ranges {
                        let mask = (1u128 << range.bit_count) - 1;
                        let _ = writeln!(
                            out,
                            "                word |= ({enc_value}u128 & {mask:#x}u128) << {};",
                            range.bit_offset
                        );
                    }
                }
                if let Some(op_field) = enc_def.fields.iter().find(|f| f.name == "OP") {
                    for range in &op_field.ranges {
                        let mask = (1u128 << range.bit_count) - 1;
                        let _ = writeln!(
                            out,
                            "                word |= ({}u128 & {mask:#x}u128) << {};",
                            enc.opcode, range.bit_offset
                        );
                    }
                }
                for (field_name, _rust_type) in &fields {
                    let orig_name = field_name.to_uppercase();
                    if let Some(mc_field) = enc_def.fields.iter().find(|f| {
                        f.name == orig_name || to_snake_case(&f.name) == *field_name
                    }) {
                        for range in &mc_field.ranges {
                            let mask = (1u128 << range.bit_count) - 1;
                            let _ = writeln!(
                                out,
                                "                word |= (*{field_name} as u128 & {mask:#x}u128) << {};",
                                range.bit_offset
                            );
                        }
                    }
                }
                let _ = writeln!(out, "                word.to_le_bytes()[..{byte_count}].to_vec()");
            }
        } else {
            let _ = writeln!(out, "                Vec::new()");
        }

        let _ = writeln!(out, "            }}");
    }

    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");

    // Mnemonic method
    let mnemonic = inst.name.to_lowercase();
    let _ = writeln!(out, "    /// SP3 instruction mnemonic.");
    let _ = writeln!(
        out,
        "    pub fn mnemonic(&self) -> &'static str {{ \"{mnemonic}\" }}"
    );

    // Encoding name method
    let _ = writeln!(out, "    /// Encoding format name.");
    let _ = writeln!(out, "    pub fn encoding_name(&self) -> &'static str {{");
    let _ = writeln!(out, "        match self {{");
    let mut used_variants2: HashMap<String, usize> = HashMap::new();
    for enc in &inst.encodings {
        let variant_name = unique_variant_name(
            &enc.encoding_name,
            &enc.encoding_condition,
            &mut used_variants2,
        );
        let fields = collect_fields(enc, encoding_map);
        if fields.is_empty() {
            let _ = writeln!(
                out,
                "            Self::{variant_name} => \"{}\",",
                enc.encoding_name
            );
        } else {
            let _ = writeln!(
                out,
                "            Self::{variant_name} {{ .. }} => \"{}\",",
                enc.encoding_name
            );
        }
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");

    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

// ─── Main instruction enum ─────────────────────────────────────────────────────

fn generate_main_enum(out: &mut String, instructions: &[InstructionDef], isa_name: &str) {
    let enum_name = format!("{isa_name}Instruction");

    let _ = writeln!(out, "/// All instructions in the {} ISA.", isa_name);
    let _ = writeln!(out, "#[derive(Debug, Clone, PartialEq)]");
    let _ = writeln!(out, "#[allow(non_camel_case_types)]");
    let _ = writeln!(out, "pub enum {enum_name} {{");

    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let _ = writeln!(out, "    /// `{}`", inst.name);
        let _ = writeln!(out, "    {type_name}({type_name}),");
    }

    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

fn generate_main_display(out: &mut String, instructions: &[InstructionDef], isa_name: &str) {
    let enum_name = format!("{isa_name}Instruction");

    let _ = writeln!(out, "impl core::fmt::Display for {enum_name} {{");
    let _ = writeln!(
        out,
        "    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{"
    );
    let _ = writeln!(out, "        match self {{");

    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let _ = writeln!(out, "            Self::{type_name}(inner) => inner.fmt(f),");
    }

    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

fn generate_instruction_trait_impl(
    out: &mut String,
    instructions: &[InstructionDef],
    isa_name: &str,
) {
    let enum_name = format!("{isa_name}Instruction");

    let _ = writeln!(out, "impl crate::Instruction for {enum_name} {{");

    // mnemonic()
    let _ = writeln!(out, "    fn mnemonic(&self) -> &'static str {{");
    let _ = writeln!(out, "        match self {{");
    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let _ = writeln!(
            out,
            "            Self::{type_name}(inner) => inner.mnemonic(),"
        );
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");

    // encoding_name()
    let _ = writeln!(out, "    fn encoding_name(&self) -> &'static str {{");
    let _ = writeln!(out, "        match self {{");
    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let _ = writeln!(
            out,
            "            Self::{type_name}(inner) => inner.encoding_name(),"
        );
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");

    // is_branch()
    let _ = writeln!(out, "    fn is_branch(&self) -> bool {{");
    let _ = writeln!(out, "        match self {{");
    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let _ = writeln!(
            out,
            "            Self::{type_name}(_) => {},",
            inst.flags.is_branch
        );
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");

    // is_program_terminator()
    let _ = writeln!(out, "    fn is_program_terminator(&self) -> bool {{");
    let _ = writeln!(out, "        match self {{");
    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let _ = writeln!(
            out,
            "            Self::{type_name}(_) => {},",
            inst.flags.is_program_terminator
        );
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");

    // encode()
    let _ = writeln!(out, "    fn encode(&self) -> Vec<u8> {{");
    let _ = writeln!(out, "        match self {{");
    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let _ = writeln!(
            out,
            "            Self::{type_name}(inner) => inner.encode(),"
        );
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");

    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

// ─── ISA struct + decode ───────────────────────────────────────────────────────

fn generate_isa_struct(
    out: &mut String,
    isa: &IsaSpec,
    isa_name: &str,
    encoding_map: &HashMap<&str, &Encoding>,
    _enc_op_map: &HashMap<(&str, u32), Vec<(usize, &str)>>,
) {
    let struct_name = format!("{isa_name}Isa");
    let enum_name = format!("{isa_name}Instruction");
    let arch_name = &isa.architecture.name;

    let _ = writeln!(out, "/// ISA definition for {arch_name}.");
    let _ = writeln!(out, "pub struct {struct_name};");
    let _ = writeln!(out);
    let _ = writeln!(out, "impl crate::Isa for {struct_name} {{");
    let _ = writeln!(out, "    type Instruction = {enum_name};");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "    fn name() -> &'static str {{ \"{arch_name}\" }}"
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "    fn decode(bytes: &[u8]) -> Result<(Self::Instruction, usize), crate::DecodeError> {{"
    );
    let _ = writeln!(
        out,
        "        {struct_name}::decode(bytes)"
    );
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    // Decode implementation
    let _ = writeln!(out, "impl {struct_name} {{");
    let _ = writeln!(
        out,
        "    /// Decode an instruction from bytes (little-endian)."
    );
    let _ = writeln!(
        out,
        "    pub fn decode(bytes: &[u8]) -> Result<({enum_name}, usize), crate::DecodeError> {{"
    );
    let _ = writeln!(out, "        if bytes.len() < 4 {{");
    let _ = writeln!(out, "            return Err(crate::DecodeError::InsufficientBytes {{ needed: 4, available: bytes.len() }});");
    let _ = writeln!(out, "        }}");
    let _ = writeln!(
        out,
        "        let w0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);"
    );
    let _ = writeln!(out);

    // Sort encodings: handle only base encodings (default condition) for decode.
    // Group instructions by (encoding_name, opcode) for the default condition.
    let mut sorted_encodings: Vec<&Encoding> = isa.encodings.iter().collect();
    sorted_encodings.sort_by_key(|e| e.order);

    for enc_def in &sorted_encodings {
        // Derive the encoding prefix value from identifiers
        let Some(enc_field) = enc_def.fields.iter().find(|f| f.name == "ENCODING") else {
            continue;
        };
        let Some(op_field) = enc_def.fields.iter().find(|f| f.name == "OP") else {
            continue;
        };

        let enc_prefix = derive_encoding_prefix(enc_def);

        // Build bit extraction info for ENCODING field
        let enc_total_bits: u32 = enc_field.ranges.iter().map(|r| r.bit_count).sum();
        let enc_offset = enc_field.ranges.first().map(|r| r.bit_offset).unwrap_or(0);
        let enc_mask = (1u64 << enc_total_bits) - 1;

        let op_total_bits: u32 = op_field.ranges.iter().map(|r| r.bit_count).sum();
        let op_offset = op_field.ranges.first().map(|r| r.bit_offset).unwrap_or(0);
        let op_mask = (1u64 << op_total_bits) - 1;

        let bit_count = enc_def.bit_count;

        // Collect all instructions that use this encoding (any condition)
        let mut opcode_instructions: Vec<(u32, &InstructionDef, &InstructionEncoding)> = Vec::new();
        for inst in &isa.instructions {
            for ienc in &inst.encodings {
                if ienc.encoding_name == enc_def.name {
                    opcode_instructions.push((ienc.opcode, inst, ienc));
                }
            }
        }
        // Deduplicate by opcode (take the default condition, or first available)
        let mut seen_opcodes: HashMap<u32, (&InstructionDef, &InstructionEncoding)> =
            HashMap::new();
        for (opcode, inst, ienc) in &opcode_instructions {
            let entry = seen_opcodes.entry(*opcode);
            match entry {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert((inst, ienc));
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    // Prefer "default" condition
                    if ienc.encoding_condition == "default" {
                        e.insert((inst, ienc));
                    }
                }
            }
        }

        if seen_opcodes.is_empty() {
            continue;
        }

        let _ = writeln!(
            out,
            "        // Encoding: {} ({}-bit, order={})",
            enc_def.name, bit_count, enc_def.order
        );

        if bit_count <= 32 {
            let _ = writeln!(
                out,
                "        if ((w0 >> {enc_offset}) & {enc_mask:#x}u32) == {enc_prefix}u32 {{"
            );
            let _ = writeln!(
                out,
                "            let op = ((w0 >> {op_offset}) & {op_mask:#x}u32) as u32;"
            );
            let _ = writeln!(out, "            match op {{");

            let mut sorted_opcodes: Vec<_> = seen_opcodes.iter().collect();
            sorted_opcodes.sort_by_key(|(op, _)| **op);

            for (opcode, (inst, ienc)) in sorted_opcodes {
                let type_name = to_pascal_case(&inst.name);
                let variant_name =
                    encoding_variant_name(&ienc.encoding_name, &ienc.encoding_condition);
                let fields = collect_fields(ienc, encoding_map);

                if fields.is_empty() {
                    let _ = writeln!(
                        out,
                        "                {opcode} => return Ok(({enum_name}::{type_name}({type_name}::{variant_name}), {})),",
                        bit_count / 8
                    );
                } else {
                    let _ = writeln!(out, "                {opcode} => {{");
                    // Extract each field from w0
                    for (field_name, rust_type) in &fields {
                        let orig_name = field_name.to_uppercase();
                        if let Some(mc_field) = enc_def.fields.iter().find(|f| {
                            f.name == orig_name || to_snake_case(&f.name) == *field_name
                        }) {
                            let _total_bits: u32 =
                                mc_field.ranges.iter().map(|r| r.bit_count).sum();
                            if mc_field.ranges.len() == 1 {
                                let r = &mc_field.ranges[0];
                                let fmask = (1u64 << r.bit_count) - 1;
                                let _ = writeln!(
                                    out,
                                    "                    let {field_name} = ((w0 >> {}) & {fmask:#x}u32) as {rust_type};",
                                    r.bit_offset
                                );
                            } else {
                                // Multi-range field
                                let _ = writeln!(
                                    out,
                                    "                    let {field_name} = {{"
                                );
                                let _ = writeln!(out, "                        let mut val: u32 = 0;");
                                let _ = writeln!(out, "                        let mut shift = 0u32;");
                                let mut sorted_ranges = mc_field.ranges.clone();
                                sorted_ranges.sort_by_key(|r| r.order);
                                for r in &sorted_ranges {
                                    let rmask = (1u64 << r.bit_count) - 1;
                                    let _ = writeln!(
                                        out,
                                        "                        val |= ((w0 >> {}) & {rmask:#x}u32) << shift;",
                                        r.bit_offset
                                    );
                                    let _ = writeln!(
                                        out,
                                        "                        shift += {};",
                                        r.bit_count
                                    );
                                }
                                let _ = writeln!(
                                    out,
                                    "                        val as {rust_type}"
                                );
                                let _ = writeln!(out, "                    }};");
                            }
                        }
                    }
                    let field_names: Vec<_> =
                        fields.iter().map(|(n, _)| n.as_str()).collect();
                    let _ = writeln!(
                        out,
                        "                    return Ok(({enum_name}::{type_name}({type_name}::{variant_name} {{ {} }}), {}));",
                        field_names.join(", "),
                        bit_count / 8
                    );
                    let _ = writeln!(out, "                }}");
                }
            }

            let _ = writeln!(out, "                _ => {{}}");
            let _ = writeln!(out, "            }}");
            let _ = writeln!(out, "        }}");
            let _ = writeln!(out);
        } else {
            // >32-bit encoding (64, 96, 128 bits): use u128
            let total_bytes = bit_count / 8;
            let _ = writeln!(out, "        if bytes.len() >= {total_bytes} {{");
            let _ = writeln!(
                out,
                "            let w1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);"
            );
            let dw_mut = if bit_count >= 96 { "mut " } else { "" };
            let _ = writeln!(
                out,
                "            let {dw_mut}dw: u128 = (w1 as u128) << 32 | (w0 as u128);"
            );
            if bit_count >= 96 {
                let _ = writeln!(
                    out,
                    "            let w2 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);"
                );
                let _ = writeln!(out, "            dw |= (w2 as u128) << 64;");
            }
            if bit_count >= 128 {
                let _ = writeln!(
                    out,
                    "            let w3 = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);"
                );
                let _ = writeln!(out, "            dw |= (w3 as u128) << 96;");
            }

            // For >32-bit encodings, the ENCODING field is typically in the lower 32 bits
            if enc_offset < 32 {
                let _ = writeln!(
                    out,
                    "            if ((w0 >> {enc_offset}) & {enc_mask:#x}u32) == {enc_prefix}u32 {{"
                );
            } else {
                let off = enc_offset;
                let _ = writeln!(
                    out,
                    "            if ((dw >> {off}) & {enc_mask:#x}u128) == {enc_prefix}u128 {{"
                );
            }

            if op_offset < 32 {
                let _ = writeln!(
                    out,
                    "                let op = ((w0 >> {op_offset}) & {op_mask:#x}u32) as u32;"
                );
            } else {
                let _ = writeln!(
                    out,
                    "                let op = ((dw >> {op_offset}) & {op_mask:#x}u128) as u32;"
                );
            }

            let _ = writeln!(out, "                match op {{");

            let mut sorted_opcodes: Vec<_> = seen_opcodes.iter().collect();
            sorted_opcodes.sort_by_key(|(op, _)| **op);

            for (opcode, (inst, ienc)) in sorted_opcodes {
                let type_name = to_pascal_case(&inst.name);
                let variant_name =
                    encoding_variant_name(&ienc.encoding_name, &ienc.encoding_condition);
                let fields = collect_fields(ienc, encoding_map);

                if fields.is_empty() {
                    let _ = writeln!(
                        out,
                        "                    {opcode} => return Ok(({enum_name}::{type_name}({type_name}::{variant_name}), {total_bytes})),"
                    );
                } else {
                    let _ = writeln!(out, "                    {opcode} => {{");
                    for (field_name, rust_type) in &fields {
                        let orig_name = field_name.to_uppercase();
                        if let Some(mc_field) = enc_def.fields.iter().find(|f| {
                            f.name == orig_name || to_snake_case(&f.name) == *field_name
                        }) {
                            if mc_field.ranges.len() == 1 {
                                let r = &mc_field.ranges[0];
                                let fmask = (1u128 << r.bit_count) - 1;
                                let _ = writeln!(
                                    out,
                                    "                        let {field_name} = ((dw >> {}) & {fmask:#x}u128) as {rust_type};",
                                    r.bit_offset
                                );
                            } else {
                                let _ = writeln!(out, "                        let {field_name} = {{");
                                let _ = writeln!(out, "                            let mut val: u128 = 0;");
                                let _ = writeln!(out, "                            let mut shift = 0u32;");
                                let mut sorted_ranges = mc_field.ranges.clone();
                                sorted_ranges.sort_by_key(|r| r.order);
                                for r in &sorted_ranges {
                                    let rmask = (1u128 << r.bit_count) - 1;
                                    let _ = writeln!(
                                        out,
                                        "                            val |= ((dw >> {}) & {rmask:#x}u128) << shift;",
                                        r.bit_offset
                                    );
                                    let _ = writeln!(
                                        out,
                                        "                            shift += {};",
                                        r.bit_count
                                    );
                                }
                                let _ = writeln!(out, "                            val as {rust_type}");
                                let _ = writeln!(out, "                        }};");
                            }
                        }
                    }
                    let field_names: Vec<_> =
                        fields.iter().map(|(n, _)| n.as_str()).collect();
                    let _ = writeln!(
                        out,
                        "                        return Ok(({enum_name}::{type_name}({type_name}::{variant_name} {{ {} }}), {total_bytes}));",
                        field_names.join(", ")
                    );
                    let _ = writeln!(out, "                    }}");
                }
            }

            let _ = writeln!(out, "                    _ => {{}}");
            let _ = writeln!(out, "                }}");
            let _ = writeln!(out, "            }}");
            let _ = writeln!(out, "        }}");
            let _ = writeln!(out);
        }
    }

    let _ = writeln!(
        out,
        "        Err(crate::DecodeError::UnknownInstruction(w0))"
    );
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

// ─── Visitor trait ─────────────────────────────────────────────────────────────

fn generate_visitor_trait(out: &mut String, instructions: &[InstructionDef], isa_name: &str) {
    let trait_name = format!("{isa_name}Visitor");
    let enum_name = format!("{isa_name}Instruction");

    let _ = writeln!(
        out,
        "/// Visitor trait for {} instructions. Implement this to handle each instruction type.",
        isa_name
    );
    let _ = writeln!(out, "pub trait {trait_name} {{");
    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let method_name = format!("visit_{}", inst.name.to_lowercase());
        let _ = writeln!(
            out,
            "    fn {method_name}(&mut self, _inst: &{type_name}) {{}}"
        );
    }
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    // dispatch method on the instruction enum
    let _ = writeln!(out, "impl {enum_name} {{");
    let _ = writeln!(
        out,
        "    /// Dispatch to the appropriate visitor method."
    );
    let _ = writeln!(
        out,
        "    pub fn accept<V: {trait_name}>(&self, visitor: &mut V) {{"
    );
    let _ = writeln!(out, "        match self {{");
    for inst in instructions {
        let type_name = to_pascal_case(&inst.name);
        let method_name = format!("visit_{}", inst.name.to_lowercase());
        let _ = writeln!(
            out,
            "            Self::{type_name}(inner) => visitor.{method_name}(inner),"
        );
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/// Convert "V_ADD_F32" → "VAddF32" (PascalCase).
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|p| !p.is_empty())
        .map(|part| {
            let mut c = part.chars();
            match c.next() {
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    format!("{upper}{}", c.as_str().to_lowercase())
                }
                None => String::new(),
            }
        })
        .collect()
}

/// Convert "VDST" → "vdst", "LANE_SEL_0" → "lane_sel_0".
fn to_snake_case(s: &str) -> String {
    // Already UPPER_CASE or UPPERCASE — just lowercase it
    s.to_lowercase()
}

/// Pick a Rust type based on bit count.
fn rust_type_for_bits(bits: u32) -> &'static str {
    match bits {
        0..=8 => "u8",
        9..=16 => "u16",
        17..=32 => "u32",
        _ => "u64",
    }
}

/// Generate a unique variant name from encoding name + condition.
/// Uses a HashMap to track and deduplicate names across an instruction's encodings.
fn unique_variant_name(
    encoding_name: &str,
    condition: &str,
    used: &mut HashMap<String, usize>,
) -> String {
    let base = encoding_variant_name(encoding_name, condition);
    let count = used.entry(base.clone()).or_insert(0);
    let name = if *count == 0 {
        base.clone()
    } else {
        format!("{base}{count}")
    };
    *used.get_mut(&base).unwrap() += 1;
    name
}

/// Generate a base variant name from encoding name + condition.
fn encoding_variant_name(encoding_name: &str, condition: &str) -> String {
    if condition == "default" || condition.is_empty() {
        to_pascal_case(encoding_name)
    } else {
        format!(
            "{}{}",
            to_pascal_case(encoding_name),
            to_pascal_case(condition)
        )
    }
}

/// Collect the (field_name, rust_type) pairs for an instruction encoding's operands.
fn collect_fields(
    enc: &InstructionEncoding,
    encoding_map: &HashMap<&str, &Encoding>,
) -> Vec<(String, &'static str)> {
    let mut fields = Vec::new();
    let mut seen = HashMap::new();

    for op in &enc.operands {
        // Skip implicit operands and operands without field names
        if op.is_implicit {
            continue;
        }
        let Some(ref fname) = op.field_name else {
            continue;
        };

        let mut rust_name = to_snake_case(fname);

        // Determine bit count from the encoding's microcode format
        let bits = if let Some(enc_def) = encoding_map.get(enc.encoding_name.as_str()) {
            enc_def
                .fields
                .iter()
                .find(|f| f.name == *fname)
                .map(|f| f.total_bits())
                .unwrap_or(op.operand_size)
        } else {
            op.operand_size
        };

        // Handle duplicate field names
        let count = seen.entry(rust_name.clone()).or_insert(0u32);
        if *count > 0 {
            rust_name = format!("{rust_name}_{count}");
        }
        *seen.get_mut(&to_snake_case(fname)).unwrap() += 1;

        let rust_type = rust_type_for_bits(bits);
        fields.push((rust_name, rust_type));
    }

    fields
}

/// Derive the encoding prefix (the ENCODING field value) from encoding identifiers.
fn derive_encoding_prefix(enc: &Encoding) -> u64 {
    let Some(enc_field) = enc.fields.iter().find(|f| f.name == "ENCODING") else {
        return 0;
    };

    // Extract the ENCODING field value from the first identifier
    if let Some(&first_id) = enc.identifiers.first() {
        let mut value: u64 = 0;
        let mut shift = 0u32;
        let mut sorted_ranges = enc_field.ranges.clone();
        sorted_ranges.sort_by_key(|r| r.order);
        for range in &sorted_ranges {
            let mask = (1u64 << range.bit_count) - 1;
            value |= ((first_id >> range.bit_offset) & mask) << shift;
            shift += range.bit_count;
        }
        value
    } else {
        0
    }
}

/// Escape text for Rust doc comments.
fn escape_doc(s: &str) -> String {
    s.replace('[', "\\[")
        .replace(']', "\\]")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
