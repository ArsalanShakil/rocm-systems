//! Tensors and a minimal, self-contained NumPy `.npy` reader/writer.
//!
//! The corpus exchanges inputs and outputs with `iree-run-module` as
//! little-endian NumPy `.npy` v1.0 files. Rather than pull in the full
//! `ndarray` stack we implement just enough of the format here: all values
//! are kept as `f64` for arithmetic, alongside the on-disk [`Dtype`] used
//! when writing.

use std::path::Path;

use crate::error::{CorpusError, Result};

/// Numeric element type of a tensor on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// Boolean (1 byte).
    Bool,
}

impl Dtype {
    /// Parse a numpy dtype string (a NumPy name like `float32`, an IREE
    /// element-type name like `f32`/`i32`, or a `.npy` descr like `<f4`).
    ///
    /// Bare short names use *bit-width* semantics (IREE/NumPy convention):
    /// `i8` is an 8-bit int, `f32` a 32-bit float. The `<`/`|`-prefixed
    /// forms are `.npy` descrs and use *byte-count* semantics (`<i8` is a
    /// 64-bit int), so the two never collide.
    pub fn parse(s: &str) -> Result<Dtype> {
        let d = match s {
            "float32" | "f32" | "<f4" | "|f4" => Dtype::F32,
            "float64" | "double" | "f64" | "<f8" | "|f8" => Dtype::F64,
            "int8" | "i8" | "<i1" | "|i1" => Dtype::I8,
            "int16" | "i16" | "<i2" | "|i2" => Dtype::I16,
            "int32" | "i32" | "<i4" | "|i4" => Dtype::I32,
            "int64" | "i64" | "<i8" | "|i8" => Dtype::I64,
            "uint8" | "u8" | "<u1" | "|u1" => Dtype::U8,
            "uint32" | "u32" | "<u4" | "|u4" => Dtype::U32,
            "uint64" | "u64" | "<u8" | "|u8" => Dtype::U64,
            "bool" | "b1" | "<b1" | "|b1" | "?" => Dtype::Bool,
            other => {
                return Err(CorpusError::other(format!(
                    "unsupported dtype {other:?} (bfloat16/float16 are not supported)"
                )));
            }
        };
        Ok(d)
    }

    /// The `.npy` `descr` string for this dtype.
    fn descr(&self) -> &'static str {
        match self {
            Dtype::F32 => "<f4",
            Dtype::F64 => "<f8",
            Dtype::I8 => "|i1",
            Dtype::I16 => "<i2",
            Dtype::I32 => "<i4",
            Dtype::I64 => "<i8",
            Dtype::U8 => "|u1",
            Dtype::U32 => "<u4",
            Dtype::U64 => "<u8",
            Dtype::Bool => "|b1",
        }
    }

    /// Bytes per element.
    fn size(&self) -> usize {
        match self {
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
            Dtype::I16 => 2,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        }
    }
}

/// A dense tensor: a shape plus row-major `f64` data and its on-disk dtype.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Row-major elements, widened to `f64`.
    pub data: Vec<f64>,
    /// The dtype used when (re)writing this tensor.
    pub dtype: Dtype,
}

impl Tensor {
    /// Build a tensor, validating that `data` matches `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f64>, dtype: Dtype) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(CorpusError::other(format!(
                "tensor shape {shape:?} ({n} elements) does not match data length {}",
                data.len()
            )));
        }
        Ok(Tensor { shape, data, dtype })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row-major strides for the shape.
    fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }

    /// Fetch the value at a multi-dimensional index, if in bounds.
    pub fn at(&self, index: &[usize]) -> Option<f64> {
        if index.len() != self.shape.len() {
            return None;
        }
        let strides = self.strides();
        let mut flat = 0usize;
        for (i, (&idx, &dim)) in index.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return None;
            }
            flat += idx * strides[i];
        }
        self.data.get(flat).copied()
    }

    /// Read a tensor from a `.npy` file.
    pub fn read_npy(path: &Path) -> Result<Tensor> {
        let bytes = std::fs::read(path).map_err(|source| CorpusError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_npy_bytes(&bytes, path)
    }

    /// Parse a tensor from in-memory `.npy` bytes.
    pub fn from_npy_bytes(bytes: &[u8], path: &Path) -> Result<Tensor> {
        let invalid = |m: &str| CorpusError::invalid(path, m.to_string());
        if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
            return Err(invalid("not a .npy file (bad magic)"));
        }
        let major = bytes[6];
        let (header_start, header_len) = if major == 1 {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (10usize, len)
        } else {
            if bytes.len() < 12 {
                return Err(invalid("truncated .npy header"));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (12usize, len)
        };
        let header_end = header_start + header_len;
        if header_end > bytes.len() {
            return Err(invalid("truncated .npy header"));
        }
        let header = std::str::from_utf8(&bytes[header_start..header_end])
            .map_err(|_| invalid("non-utf8 .npy header"))?;

        let descr = extract_field(header, "descr")
            .ok_or_else(|| invalid("missing descr in .npy header"))?;
        let dtype = Dtype::parse(descr.trim_matches(|c| c == '\'' || c == '"'))?;
        if header.contains("'fortran_order': True") {
            return Err(invalid("fortran-ordered .npy files are not supported"));
        }
        let shape = extract_shape(header).ok_or_else(|| invalid("missing shape in .npy header"))?;

        let n: usize = shape.iter().product();
        let data_bytes = &bytes[header_end..];
        if data_bytes.len() < n * dtype.size() {
            return Err(invalid("truncated .npy data section"));
        }
        let data = decode(data_bytes, dtype, n);
        Ok(Tensor { shape, data, dtype })
    }

    /// Write this tensor as a `.npy` v1.0 file.
    pub fn write_npy(&self, path: &Path) -> Result<()> {
        let bytes = self.to_npy_bytes();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| CorpusError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        std::fs::write(path, bytes).map_err(|source| CorpusError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Encode this tensor as `.npy` v1.0 bytes.
    pub fn to_npy_bytes(&self) -> Vec<u8> {
        let shape_str = if self.shape.is_empty() {
            "()".to_string()
        } else if self.shape.len() == 1 {
            format!("({},)", self.shape[0])
        } else {
            let inner: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
            format!("({})", inner.join(", "))
        };
        let header = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
            self.dtype.descr(),
            shape_str
        );
        // Pad so that 10 + header.len() + 1 (newline) is a multiple of 64.
        let unpadded = 10 + header.len() + 1;
        let pad = (64 - (unpadded % 64)) % 64;
        let mut header_bytes = header.into_bytes();
        header_bytes.extend(std::iter::repeat_n(b' ', pad));
        header_bytes.push(b'\n');

        let mut out = Vec::with_capacity(10 + header_bytes.len() + self.data.len() * 8);
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        encode(&self.data, self.dtype, &mut out);
        out
    }
}

fn extract_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("'{key}'");
    let start = header.find(&needle)? + needle.len();
    let rest = &header[start..];
    let colon = rest.find(':')? + 1;
    let rest = rest[colon..].trim_start();
    // value ends at the next comma or closing brace at depth 0
    let end = rest.find([',', '}'])?;
    Some(rest[..end].trim())
}

fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape'")?;
    let open = header[start..].find('(')? + start + 1;
    let close = header[open..].find(')')? + open;
    let inner = &header[open..close];
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(p.parse::<usize>().ok()?);
    }
    Some(shape)
}

fn decode(bytes: &[u8], dtype: Dtype, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let sz = dtype.size();
    for i in 0..n {
        let b = &bytes[i * sz..i * sz + sz];
        let v = match dtype {
            Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Dtype::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            Dtype::I8 => (b[0] as i8) as f64,
            Dtype::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            Dtype::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Dtype::I64 => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64,
            Dtype::U8 => b[0] as f64,
            Dtype::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Dtype::U64 => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64,
            Dtype::Bool => {
                if b[0] != 0 {
                    1.0
                } else {
                    0.0
                }
            }
        };
        out.push(v);
    }
    out
}

fn encode(data: &[f64], dtype: Dtype, out: &mut Vec<u8>) {
    for &v in data {
        match dtype {
            Dtype::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            Dtype::F64 => out.extend_from_slice(&v.to_le_bytes()),
            Dtype::I8 => out.push((v as i64 as i8) as u8),
            Dtype::I16 => out.extend_from_slice(&(v as i64 as i16).to_le_bytes()),
            Dtype::I32 => out.extend_from_slice(&(v as i64 as i32).to_le_bytes()),
            Dtype::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
            Dtype::U8 => out.push(v as i64 as u8),
            Dtype::U32 => out.extend_from_slice(&(v as i64 as u32).to_le_bytes()),
            Dtype::U64 => out.extend_from_slice(&(v as i64 as u64).to_le_bytes()),
            Dtype::Bool => out.push(if v != 0.0 { 1 } else { 0 }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_roundtrip_f32() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Dtype::F32).unwrap();
        let bytes = t.to_npy_bytes();
        let back = Tensor::from_npy_bytes(&bytes, Path::new("mem")).unwrap();
        assert_eq!(back.shape, vec![2, 3]);
        assert_eq!(back.dtype, Dtype::F32);
        assert_eq!(back.data, t.data);
    }

    #[test]
    fn npy_roundtrip_i32_and_index() {
        let t = Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0], Dtype::I32).unwrap();
        let bytes = t.to_npy_bytes();
        let back = Tensor::from_npy_bytes(&bytes, Path::new("mem")).unwrap();
        assert_eq!(back.at(&[1, 0]), Some(30.0));
        assert_eq!(back.at(&[0, 1]), Some(20.0));
        assert_eq!(back.at(&[2, 0]), None);
    }

    #[test]
    fn header_is_64_byte_aligned() {
        let t = Tensor::new(vec![4], vec![0.0; 4], Dtype::F64).unwrap();
        let bytes = t.to_npy_bytes();
        // 10-byte preamble + header must be a multiple of 64.
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(Tensor::from_npy_bytes(b"not npy", Path::new("x")).is_err());
    }

    #[test]
    fn dtype_parse_rejects_bf16() {
        assert!(Dtype::parse("bfloat16").is_err());
        assert!(Dtype::parse("float16").is_err());
    }
}
