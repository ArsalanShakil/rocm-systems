//! Integration tests for the amdgpu_isa crate.

use amdgpu_isa::{DecodeError, EncodingFormat, Instruction, Isa, format_src};

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
    assert_eq!(
        EncodingFormat::Sop1InstLiteral.as_str(),
        "SOP1_INST_LITERAL"
    );
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
    let xml =
        std::fs::read_to_string("data/amdgpu_isa_rdna4.xml").expect("failed to read RDNA4 XML");
    let spec = amdgpu_isa::schema::Spec::parse(&xml).expect("failed to parse RDNA4 XML");

    assert_eq!(spec.isa.architecture.name, "AMD RDNA 4");
    assert!(!spec.isa.encodings.is_empty(), "should have encodings");
    assert!(
        !spec.isa.instructions.is_empty(),
        "should have instructions"
    );
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

    let vop2 = spec
        .isa
        .encodings
        .iter()
        .find(|e| e.name == "ENC_VOP2")
        .expect("ENC_VOP2 encoding not found");

    assert_eq!(vop2.bit_count, 32);

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

// ─── RDNA4 encode/decode tests ─────────────────────────────────────────────────

#[cfg(feature = "rdna4")]
mod rdna4_tests {
    use super::*;
    use std::io::Cursor;

    // Helpers to exercise the Read/Write based API.
    fn encode_with<F: FnOnce(&mut Vec<u8>) -> std::io::Result<()>>(f: F) -> Vec<u8> {
        let mut buf = Vec::new();
        f(&mut buf).expect("encode failed");
        buf
    }

    fn decode_bytes(bytes: &[u8]) -> Result<RDNA4Instruction, DecodeError> {
        let mut cursor = Cursor::new(bytes);
        RDNA4Isa::decode(&mut cursor)
    }

    fn decode_bytes_with_position(bytes: &[u8]) -> (RDNA4Instruction, u64) {
        let mut cursor = Cursor::new(bytes);
        let inst = RDNA4Isa::decode(&mut cursor).expect("decode failed");
        (inst, cursor.position())
    }

    // ── VOP2: V_ADD_F32 ──

    #[test]
    fn test_v_add_f32_encode_decode_roundtrip() {
        let inst = VAddF32::EncVop2 {
            vdst: 5,
            src0: 256, // v0
            vsrc1: 10,
        };

        let encoded = encode_with(|w| inst.encode(w));
        assert_eq!(encoded.len(), 4, "VOP2 should be 4 bytes");

        let (decoded, consumed) = decode_bytes_with_position(&encoded);
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "v_add_f32");

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
            src0: 256,
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
        let inst = VAddF32::EncVop2 {
            vdst: 0,
            src0: 0, // s0
            vsrc1: 1,
        };
        let encoded = encode_with(|w| inst.encode(w));
        let decoded = decode_bytes(&encoded).expect("decode failed");
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
        let inst = VAddF32::EncVop2 {
            vdst: 2,
            src0: 242, // 1.0
            vsrc1: 3,
        };
        let encoded = encode_with(|w| inst.encode(w));
        let decoded = decode_bytes(&encoded).expect("decode failed");
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
        let encoded = encode_with(|w| inst.encode(w));
        assert_eq!(encoded.len(), 4, "SOPP should be 4 bytes");

        let (decoded, consumed) = decode_bytes_with_position(&encoded);
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
        let vop2 = VAddF32::EncVop2 {
            vdst: 0,
            src0: 0,
            vsrc1: 0,
        };
        assert_eq!(vop2.encoding_format(), EncodingFormat::EncVop2);

        let sopp = SEndpgm::EncSopp;
        assert_eq!(sopp.encoding_format(), EncodingFormat::EncSopp);
    }

    #[test]
    fn test_s_endpgm_known_encoding() {
        // ENCODING = 0x17F (9 bits at offset 23), OP = 48 (7 bits at offset 16)
        let expected_word: u32 = 0xBFB0_0000;
        let bytes = expected_word.to_le_bytes();

        let (decoded, consumed) = decode_bytes_with_position(&bytes);
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
        let encoded = encode_with(|w| inst.encode(w));
        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn test_isa_trait_name() {
        assert_eq!(RDNA4Isa::name(), "AMD RDNA 4");
    }

    #[test]
    fn test_isa_trait_decode() {
        let inst = SEndpgm::EncSopp;
        let encoded = encode_with(|w| inst.encode(w));
        let mut cursor = Cursor::new(&encoded);
        let result = <RDNA4Isa as Isa>::decode(&mut cursor);
        assert!(result.is_ok());
    }

    // ── Error handling ──

    #[test]
    fn test_decode_insufficient_bytes() {
        let bytes = [0u8; 3];
        let mut cursor = Cursor::new(&bytes);
        let result = RDNA4Isa::decode(&mut cursor);
        match result {
            Err(DecodeError::Io(e)) => {
                assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
            }
            other => panic!("expected Io(UnexpectedEof), got {:?}", other),
        }
    }

    #[test]
    fn test_decode_empty_input() {
        let bytes: [u8; 0] = [];
        let mut cursor = Cursor::new(&bytes);
        let result = RDNA4Isa::decode(&mut cursor);
        assert!(matches!(result, Err(DecodeError::Io(_))));
    }

    #[test]
    fn test_decode_unknown_instruction() {
        let bytes = [0u8; 4];
        let result = decode_bytes(&bytes);
        // Either an unknown-instruction error, a decoded instruction with
        // opcode 0 of some encoding, or an Io EOF from a wider encoding
        // whose prefix also matches zero.
        match result {
            Ok(_) => {}
            Err(DecodeError::UnknownInstruction(_)) => {}
            Err(DecodeError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {}
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    // ── Display ──

    #[test]
    fn test_main_enum_display() {
        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        assert_eq!(format!("{inst}"), "s_endpgm");
    }

    #[test]
    fn test_main_enum_display_with_operands() {
        let inner = VAddF32::EncVop2 {
            vdst: 3,
            src0: 256,
            vsrc1: 7,
        };
        let inst = RDNA4Instruction::VAddF32(inner);
        let display = format!("{inst}");
        assert!(
            display.contains("v_add_f32"),
            "should contain mnemonic: {display}"
        );
    }

    #[test]
    fn test_v_add_f32_has_multiple_encodings() {
        let vop2 = VAddF32::EncVop2 {
            vdst: 0,
            src0: 0,
            vsrc1: 0,
        };
        assert!(format!("{vop2}").starts_with("v_add_f32"));
        assert_eq!(vop2.mnemonic(), "v_add_f32");
    }

    // ── SOP2 ──

    #[test]
    fn test_s_add_f32_roundtrip() {
        let inst = SAddF32::EncSop2 {
            sdst: 0,
            ssrc0: 1,
            ssrc1: 2,
        };
        let encoded = encode_with(|w| inst.encode(w));
        assert_eq!(encoded.len(), 4, "SOP2 should be 4 bytes");

        let (decoded, consumed) = decode_bytes_with_position(&encoded);
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "s_add_f32");
    }

    // ── Known binary ──

    #[test]
    fn test_v_add_f32_known_binary() {
        let word: u32 = (3 << 25) | (5 << 17) | (10 << 9) | 256;
        let bytes = word.to_le_bytes();

        let (decoded, consumed) = decode_bytes_with_position(&bytes);
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

    // ── Visitor ──

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

    // ── Instruction stream decoding ──

    #[test]
    fn test_decode_instruction_stream() {
        let v_add = VAddF32::EncVop2 {
            vdst: 1,
            src0: 256,
            vsrc1: 2,
        };
        let s_end = SEndpgm::EncSopp;

        let mut stream = Vec::new();
        v_add.encode(&mut stream).unwrap();
        s_end.encode(&mut stream).unwrap();
        assert_eq!(stream.len(), 8);

        // Decode both in sequence from a single cursor
        let mut cursor = Cursor::new(&stream);
        let inst1 = RDNA4Isa::decode(&mut cursor).expect("decode first");
        assert_eq!(inst1.mnemonic(), "v_add_f32");
        assert_eq!(cursor.position(), 4);

        let inst2 = RDNA4Isa::decode(&mut cursor).expect("decode second");
        assert_eq!(inst2.mnemonic(), "s_endpgm");
        assert_eq!(cursor.position(), 8);
    }

    // ── Encode to custom writer ──

    #[test]
    fn test_encode_to_arbitrary_writer() {
        let inst = SEndpgm::EncSopp;
        // Write into a fixed-size buffer via std::io::Write
        let mut buf = [0u8; 8];
        let mut slice: &mut [u8] = &mut buf;
        inst.encode(&mut slice).expect("encode should succeed");
        // After writing 4 bytes, slice has shrunk by 4
        assert_eq!(slice.len(), 4);
        // First 4 bytes should match the known S_ENDPGM encoding
        assert_eq!(&buf[..4], &0xBFB0_0000u32.to_le_bytes());
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
        assert!(
            debug.contains("EncVop2"),
            "debug should show variant: {debug}"
        );
        assert!(
            debug.contains("vdst: 5"),
            "debug should show fields: {debug}"
        );
    }

    // ── Edge cases ──

    #[test]
    fn test_max_vgpr_values() {
        let inst = VAddF32::EncVop2 {
            vdst: 255,
            src0: 511,
            vsrc1: 255,
        };
        let encoded = encode_with(|w| inst.encode(w));
        let decoded = decode_bytes(&encoded).expect("decode failed");
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
        let encoded = encode_with(|w| inst.encode(w));
        let decoded = decode_bytes(&encoded).expect("decode failed");
        if let RDNA4Instruction::VAddF32(VAddF32::EncVop2 { vdst, src0, vsrc1 }) = &decoded {
            assert_eq!(*vdst, 0);
            assert_eq!(*src0, 0);
            assert_eq!(*vsrc1, 0);
        } else {
            panic!("expected VAddF32::EncVop2");
        }
    }

    // ── DecodeError ──

    #[test]
    fn test_decode_error_display() {
        let err = DecodeError::UnknownInstruction(0xDEADBEEF);
        assert_eq!(format!("{err}"), "unknown instruction: 0xdeadbeef");

        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let err = DecodeError::Io(io_err);
        assert!(format!("{err}").starts_with("io error:"));
    }

    #[test]
    fn test_decode_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "oops");
        let err: DecodeError = io_err.into();
        assert!(matches!(err, DecodeError::Io(_)));
    }

    // ── SOP1 ──

    #[test]
    fn test_s_mov_b32_roundtrip() {
        let inst = SMovB32::EncSop1NothasLit0NothasLit1 { sdst: 4, ssrc0: 10 };
        let encoded = encode_with(|w| inst.encode(w));
        assert_eq!(encoded.len(), 4, "SOP1 should be 4 bytes");

        let (decoded, consumed) = decode_bytes_with_position(&encoded);
        assert_eq!(consumed, 4);
        assert_eq!(decoded.mnemonic(), "s_mov_b32");

        if let RDNA4Instruction::SMovB32(SMovB32::EncSop1NothasLit0NothasLit1 { sdst, ssrc0 }) =
            &decoded
        {
            assert_eq!(*sdst, 4);
            assert_eq!(*ssrc0, 10);
        } else {
            panic!(
                "expected SMovB32::EncSop1NothasLit0NothasLit1, got {:?}",
                decoded
            );
        }
    }

    // ── Instruction flags ──

    #[test]
    fn test_s_endpgm_is_program_terminator() {
        let inst = RDNA4Instruction::SEndpgm(SEndpgm::EncSopp);
        assert!(inst.is_program_terminator());
        assert!(!inst.is_branch());
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

    // ── Branch ──

    #[test]
    fn test_s_branch_roundtrip() {
        let inst = SBranch::EncSopp { simm16: 0x100 };
        let encoded = encode_with(|w| inst.encode(w));
        assert_eq!(encoded.len(), 4);

        let (decoded, consumed) = decode_bytes_with_position(&encoded);
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
        assert!(inst.is_branch());
    }
}
