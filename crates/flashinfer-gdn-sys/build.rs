use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const FLASHINFER_VERSION: &str = "0.6.16.post2";
const FLASHINFER_GIT_REV: &str = "c498513a891d424e9ebb2518a1a3c53122dbf257";

const REQUIRED_SOURCE_FILES: &[&str] = &[
    "LICENSE",
    "flashinfer/gdn_decode.py",
    "flashinfer/gdn_prefill.py",
    "flashinfer/gdn_kernels/gdn_decode_pretranspose.py",
    "flashinfer/gdn_kernels/gdn_decode_nontranspose.py",
    "flashinfer/gdn_kernels/gdn_decode_mtp.py",
    "flashinfer/gdn_kernels/gdn_decode_bf16_state.py",
    "flashinfer/gdn_kernels/blackwell/gdn_prefill.py",
    "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm90.py",
    "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py",
    "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm90.py",
    "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py",
];

fn main() {
    println!("cargo:rerun-if-env-changed=FLASHINFER_ROOT");

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let override_root = env::var_os("FLASHINFER_ROOT");
    let (candidate, source_label) = match override_root {
        Some(path) if path.is_empty() => panic!("FLASHINFER_ROOT is set but empty"),
        Some(path) => (PathBuf::from(path), "FLASHINFER_ROOT"),
        None => (
            manifest_dir.join("vendor").join("flashinfer"),
            "Cargo-provided FlashInfer source",
        ),
    };

    let source_root = validate_source_root(candidate, source_label);
    if source_label != "FLASHINFER_ROOT" {
        validate_pinned_git_revision(&source_root);
    }

    for relative in REQUIRED_SOURCE_FILES {
        println!(
            "cargo:rerun-if-changed={}",
            source_root.join(relative).display()
        );
    }

    println!(
        "cargo:rustc-env=FLASHINFER_GDN_SOURCE_ROOT={}",
        source_root.display()
    );
    println!("cargo:rustc-env=FLASHINFER_GDN_VERSION={FLASHINFER_VERSION}");
    println!("cargo:rustc-env=FLASHINFER_GDN_GIT_REV={FLASHINFER_GIT_REV}");
    println!("cargo:metadata=source_root={}", source_root.display());
    println!("cargo:metadata=flashinfer_version={FLASHINFER_VERSION}");
    println!("cargo:metadata=flashinfer_git_rev={FLASHINFER_GIT_REV}");
}

fn validate_source_root(candidate: PathBuf, source_label: &str) -> PathBuf {
    let root = candidate.canonicalize().unwrap_or_else(|error| {
        panic!(
            "{source_label} does not point to a readable FlashInfer source tree at {}: {error}",
            candidate.display()
        )
    });

    for relative in REQUIRED_SOURCE_FILES {
        let path = root.join(relative);
        assert!(
            path.is_file(),
            "{source_label} does not look like the FlashInfer {FLASHINFER_VERSION} GDN source tree: missing {}",
            path.display()
        );
    }
    root
}

fn validate_pinned_git_revision(root: &Path) {
    if !root.join(".git").exists() {
        // Published Cargo packages contain source files, not Git metadata. The
        // crate package itself is pinned and its source digest enters the JIT key.
        return;
    }

    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to inspect the vendored FlashInfer Git revision at {}: {error}",
                root.display()
            )
        });
    assert!(
        output.status.success(),
        "failed to inspect the vendored FlashInfer Git revision at {}: {}",
        root.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let revision = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        revision.trim(),
        FLASHINFER_GIT_REV,
        "Cargo-provided FlashInfer source must be pinned to {FLASHINFER_VERSION} ({FLASHINFER_GIT_REV})"
    );
}
