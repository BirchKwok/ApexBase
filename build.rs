use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=PATH");

    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return;
    }
    if env::var_os("CARGO_FEATURE_PYTHON").is_none() {
        return;
    }
    if env::var("PROFILE").ok().as_deref() != Some("debug") {
        return;
    }

    if let Some(lib_dir) = detect_python_lib_dir() {
        // Rust test binaries run outside Python, so dyld needs an explicit rpath
        // to the interpreter's lib directory in order to load libpython.dylib.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
}

fn detect_python_lib_dir() -> Option<PathBuf> {
    python_candidates()
        .into_iter()
        .find_map(|python| query_lib_dir(&python))
}

fn python_candidates() -> Vec<OsString> {
    let mut candidates = Vec::new();
    if let Some(python) = env::var_os("PYO3_PYTHON") {
        candidates.push(python);
    }
    if let Some(python) = env::var_os("PYTHON_SYS_EXECUTABLE") {
        candidates.push(python);
    }
    candidates.push(OsString::from("python"));
    candidates.push(OsString::from("python3"));
    candidates
}

fn query_lib_dir(python: &OsString) -> Option<PathBuf> {
    let output = Command::new(python)
        .args([
            "-c",
            "import os, sys, sysconfig; print(sysconfig.get_config_var('LIBDIR') or os.path.join(sys.prefix, 'lib'))",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let lib_dir = String::from_utf8(output.stdout).ok()?;
    let lib_dir = PathBuf::from(lib_dir.trim());
    if lib_dir.is_dir() {
        Some(lib_dir)
    } else {
        None
    }
}
