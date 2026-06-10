//! Thin helpers around [`cel_interpreter`].
//!
//! CEL is used in two places:
//!
//! * **Input generation** ([`crate::generate`]): a case's `formula`
//!   expression evaluates to a *map* describing a tensor generator (its
//!   distribution, shape, and parameters). [`eval_to_json`] runs the
//!   expression and converts the resulting CEL value into a
//!   [`serde_json::Value`] the generator can deserialize.
//! * **Output validation** ([`crate::validate`]): a case's `validate`
//!   expressions evaluate to booleans over a set of precomputed scalar
//!   *facts* about the observed and expected tensors.

use cel_interpreter::objects::Key;
use cel_interpreter::{Context, Program, Value};

use crate::error::{CorpusError, Result};

/// Build a fresh CEL context seeded with named variables.
///
/// Each `(name, value)` is a JSON value injected as a CEL variable; this
/// covers both the case `params` table and the validation facts.
pub fn context_with<'a>(vars: &[(String, serde_json::Value)]) -> Result<Context<'a>> {
    let mut ctx = Context::default();
    for (name, value) in vars {
        ctx.add_variable(name.clone(), value.clone())
            .map_err(|e| CorpusError::cel(name.clone(), format!("{e:?}")))?;
    }
    Ok(ctx)
}

/// Compile and evaluate a CEL expression against a context.
pub fn eval(label: &str, expr: &str, ctx: &Context) -> Result<Value> {
    let program =
        Program::compile(expr).map_err(|e| CorpusError::cel(label, format!("compile: {e}")))?;
    program
        .execute(ctx)
        .map_err(|e| CorpusError::cel(label, format!("execute: {e}")))
}

/// Evaluate a CEL expression and convert the result to JSON.
pub fn eval_to_json(label: &str, expr: &str, ctx: &Context) -> Result<serde_json::Value> {
    let value = eval(label, expr, ctx)?;
    Ok(value_to_json(&value))
}

/// Evaluate a CEL expression that must yield a boolean.
pub fn eval_bool(label: &str, expr: &str, ctx: &Context) -> Result<bool> {
    match eval(label, expr, ctx)? {
        Value::Bool(b) => Ok(b),
        other => Err(CorpusError::cel(
            label,
            format!("expected a boolean result, got {other:?}"),
        )),
    }
}

/// Convert a CEL [`Value`] into a [`serde_json::Value`].
pub fn value_to_json(value: &Value) -> serde_json::Value {
    use serde_json::Value as J;
    match value {
        Value::Int(i) => J::from(*i),
        Value::UInt(u) => J::from(*u),
        Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(J::Number)
            .unwrap_or(J::Null),
        Value::String(s) => J::String(s.as_ref().clone()),
        Value::Bool(b) => J::Bool(*b),
        Value::Bytes(b) => J::Array(b.iter().map(|&x| J::from(x)).collect()),
        Value::Null => J::Null,
        Value::List(items) => J::Array(items.iter().map(value_to_json).collect()),
        Value::Map(map) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in map.map.iter() {
                obj.insert(key_to_string(k), value_to_json(v));
            }
            J::Object(obj)
        }
        // Functions/durations/timestamps have no meaningful JSON form here.
        other => J::String(format!("{other:?}")),
    }
}

fn key_to_string(key: &Key) -> String {
    match key {
        Key::Int(i) => i.to_string(),
        Key::Uint(u) => u.to_string(),
        Key::Bool(b) => b.to_string(),
        Key::String(s) => s.as_ref().clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_arithmetic_with_params() {
        let ctx = context_with(&[
            ("rows".into(), serde_json::json!(4)),
            ("cols".into(), serde_json::json!(3)),
        ])
        .unwrap();
        let v = eval("test", "rows * cols", &ctx).unwrap();
        assert_eq!(value_to_json(&v), serde_json::json!(12));
    }

    #[test]
    fn eval_map_to_json() {
        let ctx = context_with(&[("n".into(), serde_json::json!(4))]).unwrap();
        let j = eval_to_json(
            "formula",
            "{'dist': 'uniform', 'shape': [n, n], 'low': 0.0, 'high': 1.0}",
            &ctx,
        )
        .unwrap();
        assert_eq!(j["dist"], serde_json::json!("uniform"));
        assert_eq!(j["shape"], serde_json::json!([4, 4]));
        assert_eq!(j["high"], serde_json::json!(1.0));
    }

    #[test]
    fn eval_bool_predicate() {
        let ctx = context_with(&[
            ("max_abs_diff".into(), serde_json::json!(0.001)),
            ("allclose".into(), serde_json::json!(true)),
        ])
        .unwrap();
        assert!(eval_bool("validate", "allclose && max_abs_diff < 0.01", &ctx).unwrap());
        assert!(!eval_bool("validate", "max_abs_diff > 1.0", &ctx).unwrap());
    }
}
