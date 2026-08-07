use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::{Error, Result};

// These intentionally match deepgemm-rs exactly. Its offset differs from the
// nominal FNV-1a 64 offset basis, so do not replace it with an off-the-shelf FNV
// implementation without preserving this value.
const FNV1A64_OFFSET: u64 = 1_469_598_103_934_665_603;
const FNV1A64_PRIME: u64 = 1_099_511_628_211;

pub(crate) fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let value = serde_json::to_value(value).map_err(|error| {
        Error::InvalidInput(format!("failed to serialize canonical cache key: {error}"))
    })?;
    serde_json::to_vec(&sort_json(value)).map_err(|error| {
        Error::InvalidInput(format!("failed to encode canonical cache key: {error}"))
    })
}

fn sort_json(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(sort_json).collect()),
        Value::Object(values) => {
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key, sort_json(value));
            }
            Value::Object(sorted)
        }
        scalar => scalar,
    }
}

pub(crate) fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a64_update(FNV1A64_OFFSET, bytes))
}

pub(crate) fn digest_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .map_err(|error| Error::io(format!("failed to open {}", path.display()), error))?;
    let mut digest = FNV1A64_OFFSET;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| Error::io(format!("failed to read {}", path.display()), error))?;
        if count == 0 {
            break;
        }
        digest = fnv1a64_update(digest, &buffer[..count]);
    }
    Ok(format!("{digest:016x}"))
}

pub(crate) fn verify_digest(path: &Path, expected: &str) -> Result<()> {
    if !is_digest(expected) {
        return Err(Error::InvalidManifest {
            path: path.to_path_buf(),
            message: format!("invalid FNV-1a 64 digest {expected:?}"),
        });
    }
    let actual = digest_file(path)?;
    if actual != expected {
        return Err(Error::DigestMismatch {
            path: path.to_path_buf(),
            expected: expected.to_owned(),
            actual,
        });
    }
    Ok(())
}

pub(crate) fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = File::create(path)
        .map_err(|error| Error::io(format!("failed to create {}", path.display()), error))?;
    serde_json::to_writer_pretty(&mut file, value).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })?;
    file.write_all(b"\n")
        .map_err(|error| Error::io(format!("failed to write {}", path.display()), error))?;
    file.sync_all()
        .map_err(|error| Error::io(format!("failed to sync {}", path.display()), error))?;
    Ok(())
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn is_digest(value: &str) -> bool {
    value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn canonical_json_sorts_nested_object_keys() {
        let left = json!({"z": 1, "a": {"y": 2, "b": 3}});
        let right = json!({"a": {"b": 3, "y": 2}, "z": 1});
        assert_eq!(
            canonical_json(&left).unwrap(),
            canonical_json(&right).unwrap()
        );
    }

    #[test]
    fn digest_matches_deepgemm_fnv1a64() {
        assert_eq!(digest_bytes(b""), "14650fb0739d0383");
        assert_eq!(digest_bytes(b"hello"), "005a0d15131ec7a1");
        assert_eq!(digest_bytes(b"DeepGEMM"), "deba39f83bb027bb");
    }
}
