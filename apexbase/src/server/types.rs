//! Type mapping between Arrow/ApexBase types and PostgreSQL types

use arrow::datatypes::DataType as ArrowDataType;
use pgwire::api::results::{FieldFormat, FieldInfo};
use pgwire::api::Type;

/// Map an Arrow DataType to a PostgreSQL Type
pub fn arrow_to_pg_type(dt: &ArrowDataType) -> Type {
    match dt {
        ArrowDataType::Boolean => Type::BOOL,
        ArrowDataType::Int8 | ArrowDataType::UInt8 => Type::INT2,
        ArrowDataType::Int16 | ArrowDataType::UInt16 => Type::INT2,
        ArrowDataType::Int32 | ArrowDataType::UInt32 => Type::INT4,
        ArrowDataType::Int64 | ArrowDataType::UInt64 => Type::INT8,
        ArrowDataType::Float16 | ArrowDataType::Float32 => Type::FLOAT4,
        ArrowDataType::Float64 => Type::FLOAT8,
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => Type::VARCHAR,
        ArrowDataType::Binary | ArrowDataType::LargeBinary => Type::BYTEA,
        ArrowDataType::Date32 | ArrowDataType::Date64 => Type::DATE,
        ArrowDataType::Timestamp(_, _) => Type::TIMESTAMP,
        ArrowDataType::Dictionary(_, value_type) => arrow_to_pg_type(value_type),
        _ => Type::VARCHAR, // fallback to text
    }
}

/// Build a list of FieldInfo from an Arrow Schema
pub fn schema_to_field_info(schema: &arrow::datatypes::Schema) -> Vec<FieldInfo> {
    schema
        .fields()
        .iter()
        .map(|field| {
            let pg_type = arrow_to_pg_type(field.data_type());
            FieldInfo::new(
                field.name().clone(),
                None,
                None,
                pg_type,
                FieldFormat::Text,
            )
        })
        .collect()
}
