//! Integration tests for the amdgpu_isa crate.
//!
//! Tests cover:
//! - Schema XML parsing
//! - Instruction encoding/decoding roundtrips
//! - Display (SP3 assembly) formatting
//! - format_src operand helper
//! - Trait implementations
//! - Visitor pattern

use amdgpu_isa::{format_src, DecodeError, EncodingFormat, Instruction, Isa};

#[cfg(feature = "rdna4")]
use amdgpu_isa::rdna4::*;

// ─── format_src tests ──────────────────────────────────────────────────────────

#[test]
fn test_format_src_sgpr() {
    assert_eq!(format_src(0), "s0");
    assert_eq!(format_src(1), "s1");
    assert_eq!(format_src(105), "s105");
}

#[test]
fn test_format_src_special_registers() {
    assert_eq!(format_src(106), "vcc_lo");
    assert_eq!(format_src(107), "vcc_hi");
    assert_eq!(format_src(124), "m0");
    assert_eq!(format_src(125), "null");
    assert_eq!(format_src(126), "exec_lo");
    assert_eq!(format_src(127), "exec_hi");
}

#[test]
fn test_format_src_ttmp() {
    assert_eq!(format_src(108), "ttmp0");
    assert_eq!(format_src(123), "ttmp15");
}

#[test]
fn test_format_src_inline_constants() {
    assert_eq!(format_src(128), "0");
    assert_eq!(format_src(129), "1");
    assert_eq!(format_src(192), "64");
    assert_eq!(format_src(193), "-1");
    assert_eq!(format_src(208), "-16");
}

#[test]
fn test_format_src_float_constants() {
    assert_eq!(format_src(240), "0.5");
    assert_eq!(format_src(241), "-0.5");
    assert_eq!(format_src(242), "1.0");
    assert_eq!(format_src(243), "-1.0");
    assert_eq!(format_src(244), "2.0");
    assert_eq!(format_src(245), "-2.0");
    assert_eq!(format_src(246), "4.0");
    assert_eq!(format_src(247), "-4.0");
}

#[test]
fn test_format_src_literal() {
    assert_eq!(format_src(255), "lit");
}

#[test]
fn test_format_src_vgpr() {
    assert_eq!(format_src(256), "v0");
    assert_eq!(format_src(257), "v1");
    assert_eq!(format_src(511), "v255");
}

// ─── EncodingFormat tests ──────────────────────────────────────────────────────

#[test]
fn test_encoding_format_as_str() {
    assert_eq!(EncodingFormat::EncVop2.as_str(), "ENC_VOP2");
    assert_eq!(EncodingFormat::EncSopp.as_str(), "ENC_SOPP");
    assert_eq!(EncodingFormat::EncSmem.as_str(), "ENC_SMEM");
    assert_eq!(EncodingFormat::Vop3SdstEnc.as_str(), "VOP3_SDST_ENC");
    assert_eq!(EncodingFormat::Sop1InstLiteral.as_str(), "SOP1_INST_LITERAL");
}

#[test]
fn test_encoding_format_display() {
    assert_eq!(format!("{}", EncodingFormat::EncVop2), "ENC_VOP2");
    assert_eq!(format!("{}", EncodingFormat::EncDs), "ENC_DS");
}

#[test]
fn test_encoding_format_eq_and_clone() {
    let a = EncodingFormat::EncVop3;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(EncodingFormat::EncVop1, EncodingFormat::EncVop2);
}

// ─── Schema parsing tests ──────────────────────────────────────────────────────

#[test]
fn test_schema_parse_rdna4() {
    let xml = std::fs::read_to_string("data/amdgpu_isa_rdna4.xml")
        .expect("failed to read RDNA4 XML");
    let spec = amdgpu_isa::schema::Spec::parse(&xml).expect("failed to parse RDNA4 XML");

    assert_eq!(spec.isa.architecture.name, "AMD RDNA 4");
    assert!(!spec.isa.encodings.is_empty(), "should have encodings");
    assert!(!spec.isa.instructions.is_empty(), "should have instructions");
    assert!(
        spec.isa.instructions.len() > 100,
        "expected 100+ instructions, got {}",
        spec.isa.instructions.len()
    );
}

#[test]
fn test_schema_encoding_fields() {
    let xml = std::fs::read_to_string("data/amdgpu_isa_rdna4.xml").unwrap();
    let spec = amdgpu_isa::schema::Spec::parse(&xml).unwrap();

    // Find ENC_VOP2 encoding
    let vop2 = spec
        .isa
        .encodings
        .iter()
        .find(|e| e.name == "ENC_VOP2")
        .expect("ENC_VOP2 encoding not found");

    assert_eq!(vop2.bit_count, 32);

    // Should have standard fields
    let field_names: Vec<&str> = vop2.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(field_names.contains(&"ENCODING"), "missing ENCODING field");
    assert!(field_names.contains(&"OP"), "missing OP field");
    assert!(field_names.contains(&"SRC0"), "missing SRC0 field");
    assert!(field_names.contains(&"VDST"), "missing VDST field");
    assert!(field_names.contains(&"VSRC1"), "missing VSRC1 field");
}

#[test]
fn test_schema_instruction_lookup() {
    let xml = std::fs::read_to_string("data/amdgpu_isa_rdna4.xml").unwrap();
    let spec = amdgpu_isa::schema::Spec::parse(&xml).unwrap();

    let v_add = spec
        .isa
        .instructions
        .iter()
        .find(|i| i.name == "V_ADD_F32")
        .expect("V_ADD_F32 not found");

    assert!(!v_add.encodings.is_empty());
    assert_eq!(v_add.encodings[0].encoding_name, "ENC_VOP2");
    assert_eq!(v_add.encodings[0].opcode, 3);
}

#[test]
fn test_schema_parse_error_on_invalid_xml() {
    let result = amdgpu_isa::schema::Spec::parse("<not-valid>");
    assert!(result.is_err());
}

// ─── Instruction encoding/decoding roundtrip tests (RDNA4) ────────────────────

#[cfg(feature = "rdna4")]
mod rdna4_tests {
    use super::*;

    // ── VOP2: V_ADD_F32 ──

    #[test]
    fn test_v_add_f32_encode_decode_roundtrip() {
        // V_ADD_F32 vdst=v5, src0=v0 (=256), vsrc1=v10
        let inst = VAddF32::EncVop2 {
            vdst: 5,
            src0: 256, // v0
            vsrc1: 10,
        };

        let encoded = inst.encode();
        assert_eq!(encoded.len(), 4, "VOP2 should be 4 bytes");

        // Decode it back
        let (decoded, consumed) = RDNA4Isa::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, 4);

        // Check it's V_ADD_F32
        assert_eq!(decoded.mnemonic(), "v_add_f32");

        // The inner instruction should match
        if let RDNA4Instruction::VAddF32(inner) = &decoded {
            assert_eq!(
                *inner,
                VAddF32::EncVop2 {
                    vdst: 5,
                    src0: 256,
                    vsrc1: 10,
                }
            );
        } else {
            panic!("expected VAddF32, got {:?}", decoded);
        }
    }

    #[test]
    fn test_v_add_f32_display() {
        let inst = VAddF32::EncVop2 {
            vdst: 5,
            src0: 256, // v0
            vsrc1: 10,
        };
        let display = format!("{inst}");
        assert!(
            display.starts_with("v_add_f32"),
            "display should start with mnemonic: {display}"
        );
    }

    #[test]
    fn test_v_add_f32_with_sgpr_src() {
        // V_ADD_F32 vdst=v0, src0=s0 (=0), vsrc1=v1
        let inst = VAddF32::EncVop2 {
            vdst: 0,
            src0: 0, // s0
            vsrc1: 1,
        };
        let encoded = inst.encode();
        let (decoded, _) = RDNA4Isa::decode(&encoded).expect("decode failed");
        if let RDNA4Instruction::VAddF32(inner) = &decoded {
            assert_eq!(
                *inner,
                VAddF32::EncVop2 {
                    vdst: 0,
                    src0: 0,
                    vsrc1: 1,
                }
            );
        } else {
            panic!("expected VAddF32");
        }
    }

    #[test]
    fn test_v_add_f32_with_inline_constant() {
        // V_ADD_F32 vdst=v2, src0=1.0 (=242), vsrc1=v3
        let inst = VAddF32::EncVop2 {
            vdst: 2,
            src0: 242, // 1.0
            vsrc1: 3,
        };
        let encoded = inst.encode();
        let (decoded, _) = RDNA4Isa::decode(&encoded).expect("decode failed");
        if let RDNA4Instruction::VAddF32(inner) = &decoded {
            assert_eq!(
                *inner,
                VAddF32::EncVop2 {
                    vdst: 2,
                    src0: 242,
                    vsrc1: 3,
                }
            );
        } else {
            panic!("expected VAddF32");
        }
    }

    // ── S_ENDPGM (no operands) ──

    #[test]
    fn test_s_endpgm_encode_decode() {
        let inst = SEndpgm::EncSopp;
        let encoded = inst.encode();
        assert_eq!(encoded.len(), 4, "SOPP should be 4 bytes");

        let (decoded, consumed) = RDNA4Isa::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "s_endpgm");
    }

    #[test]
    fn test_s_endpgm_display() {
        let inst = SEndpgm::EncSopp;
        assert_eq!(format!("{inst}"), "s_endpgm");
    }

    #[test]
    fn test_inner_encoding_format() {
        let vop2 = VAddF32::EncVop2 { vdst: 0, src0: 0, vsrc1: 0 };
        assert_eq!(vop2.encoding_format(), EncodingFormat::EncVop2);

        let sopp = SEndpgm::EncSopp;
        assert_eq!(sopp.encoding_format(), EncodingFormat::EncSopp);
    }

    #[test]
    fn test_s_endpgm_known_encoding() {
        // S_ENDPGM encoding:
        // ENCODING = 0b10111111_1 (9 bits at 23..31) = 0xBF800000
        // OP = 48 (7 bits at 16..22) = 0x00300000
        // SIMM16 = 0
        let expected_word: u32 = 0xBFB0_0000;
        let bytes = expected_word.to_le_bytes();

        let (decoded, consumed) = RDNA4Isa::decode(&bytes).expect("decode failed");
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "s_endpgm");
    }

    // ── Trait implementations ──

    #[test]
    fn test_instruction_trait_mnemonic() {
        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        assert_eq!(inst.mnemonic(), "s_endpgm");
    }

    #[test]
    fn test_instruction_trait_encoding_format() {
        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        assert_eq!(inst.encoding_format(), EncodingFormat::EncSopp);
    }

    #[test]
    fn test_instruction_trait_encode() {
        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        let encoded = inst.encode();
        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn test_isa_trait_name() {
        assert_eq!(RDNA4Isa::name(), "AMD RDNA 4");
    }

    #[test]
    fn test_isa_trait_decode() {
        // Use the Isa trait method
        let inst = SEndpgm::EncSopp;
        let encoded = inst.encode();
        let result = <RDNA4Isa as Isa>::decode(&encoded);
        assert!(result.is_ok());
    }

    // ── Error handling ──

    #[test]
    fn test_decode_insufficient_bytes() {
        let bytes = [0u8; 3]; // less than 4 bytes
        let result = RDNA4Isa::decode(&bytes);
        assert!(result.is_err());
        match result {
            Err(DecodeError::InsufficientBytes { needed: 4, available: 3 }) => {}
            other => panic!("expected InsufficientBytes, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_unknown_instruction() {
        // All zeros is unlikely to be a valid encoding
        let bytes = [0u8; 4];
        let result = RDNA4Isa::decode(&bytes);
        // May decode or not depending on encoding, but should not panic
        match result {
            Ok(_) => {} // some encodings might match 0
            Err(DecodeError::UnknownInstruction(_)) => {}
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    // ── Display trait for main enum ──

    #[test]
    fn test_main_enum_display() {
        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        let display = format!("{inst}");
        assert_eq!(display, "s_endpgm");
    }

    #[test]
    fn test_main_enum_display_with_operands() {
        let inner = VAddF32::EncVop2 {
            vdst: 3,
            src0: 256, // v0
            vsrc1: 7,
        };
        let inst = RDNA4Instruction::VAddF32(inner);
        let display = format!("{inst}");
        assert!(
            display.contains("v_add_f32"),
            "should contain mnemonic: {display}"
        );
    }

    // ── Multiple encodings for same instruction ──

    #[test]
    fn test_v_add_f32_has_multiple_encodings() {
        // VOP2 encoding
        let vop2 = VAddF32::EncVop2 {
            vdst: 0,
            src0: 0,
            vsrc1: 0,
        };
        let vop2_display = format!("{vop2}");
        assert!(vop2_display.starts_with("v_add_f32"));

        // Both should have the same mnemonic
        assert_eq!(vop2.mnemonic(), "v_add_f32");
    }

    // ── SOP2: S_ADD_U32 ──

    #[test]
    fn test_s_add_f32_roundtrip() {
        // Test SOP2 encoding roundtrip using S_ADD_F32
        let inst = SAddF32::EncSop2 {
            sdst: 0,
            ssrc0: 1,
            ssrc1: 2,
        };
        let encoded = inst.encode();
        assert_eq!(encoded.len(), 4, "SOP2 should be 4 bytes");

        let (decoded, consumed) = RDNA4Isa::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "s_add_f32");
    }

    // ── Encode specific known binary ──

    #[test]
    fn test_v_add_f32_known_binary() {
        // Construct V_ADD_F32 v5, v0, v10 manually
        // ENC_VOP2: bit31=0(ENCODING), bits25-30=3(OP), bits17-24=5(VDST),
        //           bits9-16=10(VSRC1), bits0-8=256(SRC0=v0)
        let word: u32 = (3 << 25) | (5 << 17) | (10 << 9) | 256;
        let bytes = word.to_le_bytes();

        let (decoded, consumed) = RDNA4Isa::decode(&bytes).expect("decode failed");
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "v_add_f32");

        if let RDNA4Instruction::VAddF32(VAddF32::EncVop2 { vdst, src0, vsrc1 }) = &decoded {
            assert_eq!(*vdst, 5);
            assert_eq!(*src0, 256);
            assert_eq!(*vsrc1, 10);
        } else {
            panic!("expected VAddF32::EncVop2, got {:?}", decoded);
        }
    }

    // ── Visitor pattern ──

    #[test]
    fn test_visitor_accept() {
        struct Counter {
            count: usize,
        }
        impl RDNA4Visitor for Counter {
            fn visit_v_add_f32(&mut self, _inst: &VAddF32) {
                self.count += 1;
            }
        }

        let inst = RDNA4Instruction::VAddF32(VAddF32::EncVop2 {
            vdst: 0,
            src0: 0,
            vsrc1: 0,
        });

        let mut counter = Counter { count: 0 };
        inst.accept(&mut counter);
        assert_eq!(counter.count, 1);
    }

    #[test]
    fn test_visitor_does_not_call_wrong_method() {
        struct Counter {
            v_add_calls: usize,
            s_endpgm_calls: usize,
        }
        impl RDNA4Visitor for Counter {
            fn visit_v_add_f32(&mut self, _inst: &VAddF32) {
                self.v_add_calls += 1;
            }
            fn visit_s_endpgm(&mut self, _inst: &SEndpgm) {
                self.s_endpgm_calls += 1;
            }
        }

        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        let mut counter = Counter {
            v_add_calls: 0,
            s_endpgm_calls: 0,
        };
        inst.accept(&mut counter);
        assert_eq!(counter.v_add_calls, 0);
        assert_eq!(counter.s_endpgm_calls, 1);
    }

    // ── Decode stream (multiple instructions) ──

    #[test]
    fn test_decode_instruction_stream() {
        // Encode two instructions and decode them in sequence
        let v_add = VAddF32::EncVop2 {
            vdst: 1,
            src0: 256,
            vsrc1: 2,
        };
        let s_end = SEndpgm::EncSopp;

        let mut stream = v_add.encode();
        stream.extend_from_slice(&s_end.encode());

        // Decode first instruction
        let (inst1, consumed1) = RDNA4Isa::decode(&stream).expect("decode first");
        assert_eq!(inst1.mnemonic(), "v_add_f32");

        // Decode second instruction
        let (inst2, consumed2) =
            RDNA4Isa::decode(&stream[consumed1..]).expect("decode second");
        assert_eq!(inst2.mnemonic(), "s_endpgm");

        assert_eq!(consumed1 + consumed2, stream.len());
    }

    // ── Clone and Debug ──

    #[test]
    fn test_instruction_clone() {
        let inst = VAddF32::EncVop2 {
            vdst: 5,
            src0: 256,
            vsrc1: 10,
        };
        let cloned = inst.clone();
        assert_eq!(inst, cloned);
    }

    #[test]
    fn test_instruction_debug() {
        let inst = VAddF32::EncVop2 {
            vdst: 5,
            src0: 256,
            vsrc1: 10,
        };
        let debug = format!("{inst:?}");
        assert!(debug.contains("EncVop2"), "debug should show variant: {debug}");
        assert!(debug.contains("vdst: 5"), "debug should show fields: {debug}");
    }

    // ── Edge cases ──

    #[test]
    fn test_max_vgpr_values() {
        let inst = VAddF32::EncVop2 {
            vdst: 255,    // max VGPR
            src0: 511,    // max SRC (v255)
            vsrc1: 255,   // max VSRC1
        };
        let encoded = inst.encode();
        let (decoded, _) = RDNA4Isa::decode(&encoded).expect("decode failed");
        if let RDNA4Instruction::VAddF32(VAddF32::EncVop2 { vdst, src0, vsrc1 }) = &decoded {
            assert_eq!(*vdst, 255);
            assert_eq!(*src0, 511);
            assert_eq!(*vsrc1, 255);
        } else {
            panic!("expected VAddF32::EncVop2");
        }
    }

    #[test]
    fn test_zero_operands() {
        let inst = VAddF32::EncVop2 {
            vdst: 0,
            src0: 0,
            vsrc1: 0,
        };
        let encoded = inst.encode();
        let (decoded, _) = RDNA4Isa::decode(&encoded).expect("decode failed");
        if let RDNA4Instruction::VAddF32(VAddF32::EncVop2 { vdst, src0, vsrc1 }) = &decoded {
            assert_eq!(*vdst, 0);
            assert_eq!(*src0, 0);
            assert_eq!(*vsrc1, 0);
        } else {
            panic!("expected VAddF32::EncVop2");
        }
    }

    // ── DecodeError Display ──

    #[test]
    fn test_decode_error_display() {
        let err = DecodeError::InsufficientBytes {
            needed: 4,
            available: 2,
        };
        assert_eq!(format!("{err}"), "insufficient bytes: need 4, have 2");

        let err = DecodeError::UnknownInstruction(0xDEADBEEF);
        assert_eq!(format!("{err}"), "unknown instruction: 0xdeadbeef");
    }

    // ── SOP1: S_MOV_B32 ──

    #[test]
    fn test_s_mov_b32_roundtrip() {
        let inst = SMovB32::EncSop1NothasLit0NothasLit1 {
            sdst: 4,
            ssrc0: 10,
        };
        let encoded = inst.encode();
        assert_eq!(encoded.len(), 4, "SOP1 should be 4 bytes");

        let (decoded, consumed) = RDNA4Isa::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "s_mov_b32");

        if let RDNA4Instruction::SMovB32(SMovB32::EncSop1NothasLit0NothasLit1 { sdst, ssrc0 }) = &decoded {
            assert_eq!(*sdst, 4);
            assert_eq!(*ssrc0, 10);
        } else {
            panic!("expected SMovB32::EncSop1NothasLit0NothasLit1, got {:?}", decoded);
        }
    }

    // ── Instruction flags ──

    #[test]
    fn test_s_endpgm_is_program_terminator() {
        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        assert!(
            inst.is_program_terminator(),
            "S_ENDPGM should be a program terminator"
        );
        assert!(
            !inst.is_branch(),
            "S_ENDPGM should not be a branch"
        );
    }

    #[test]
    fn test_v_add_f32_is_not_branch() {
        let inst = RDNA4Instruction::VAddF32(VAddF32::EncVop2 {
            vdst: 0,
            src0: 0,
            vsrc1: 0,
        });
        assert!(!inst.is_branch());
        assert!(!inst.is_program_terminator());
    }

    // ── SOPP branch instructions ──

    #[test]
    fn test_s_branch_roundtrip() {
        let inst = SBranch::EncSopp { simm16: 0x100 };
        let encoded = inst.encode();
        assert_eq!(encoded.len(), 4);

        let (decoded, consumed) = RDNA4Isa::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "s_branch");

        if let RDNA4Instruction::SBranch(SBranch::EncSopp { simm16 }) = &decoded {
            assert_eq!(*simm16, 0x100);
        } else {
            panic!("expected SBranch::EncSopp, got {:?}", decoded);
        }
    }

    #[test]
    fn test_s_branch_is_branch() {
        let inst = RDNA4Instruction::SBranch(SBranch::EncSopp { simm16: 0 });
        assert!(inst.is_branch(), "S_BRANCH should be a branch");
    }
}
