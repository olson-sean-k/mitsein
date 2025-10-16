#![cfg(feature = "schemars")]

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde_json::Number;
use serde_json::value::Value;

pub const NON_EMPTY_KEY_ARRAY: &str = "minItems";
pub const NON_EMPTY_KEY_OBJECT: &str = "minProperties";
// In JSON schemata, the `minLength` property is expressed in code points, which is independent of
// encoding (like UTF-8 or UTF-16). This is compatible with the non-empty definition for `Str1` and
// `String1`.
pub const NON_EMPTY_KEY_STRING: &str = "minLength";

pub fn non_empty_value() -> Value {
    Value::Number(Number::from(1u64))
}

pub fn json_subschema_with_non_empty_property_for<T>(
    key: &'static str,
    generator: &mut SchemaGenerator,
) -> Schema
where
    T: JsonSchema + ?Sized,
{
    let mut schema = generator.subschema_for::<T>();
    schema.insert(key.into(), self::non_empty_value());
    schema
}

#[cfg(test)]
pub mod harness {
    use schemars::{JsonSchema, SchemaGenerator};
    use serde_json::value::Value;

    pub fn assert_json_schema_has_non_empty_property<T>(key: &'static str)
    where
        T: JsonSchema + ?Sized,
    {
        let generator = SchemaGenerator::default();
        let schema = generator.into_root_schema_for::<T>();
        assert!(
            schema
                .get(key)
                .and_then(Value::as_u64)
                .is_some_and(|min| min != 0)
        );
    }
}
