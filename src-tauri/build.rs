use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Stage GPUPixel shared libs + VNN/lookup assets for the active target.
fn stage_gpupixel_runtime(manifest_dir: &Path, target_os: &str) {
    let desktop_src = manifest_dir.join("resources").join("gpupixel");
    if target_os == "android" {
        let abi_dir = manifest_dir.join("libs").join("arm64-v8a");
        let jni = manifest_dir.join("gen/android/app/src/main/jniLibs/arm64-v8a");
        let assets = manifest_dir.join("gen/android/app/src/main/assets/gpupixel");
        if abi_dir.is_dir() {
            let _ = fs::create_dir_all(&jni);
            let _ = fs::create_dir_all(&assets);
            if let Ok(entries) = fs::read_dir(&abi_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.ends_with(".so") {
                        let _ = fs::copy(&path, jni.join(&name));
                    } else if name_str.ends_with(".vnnmodel")
                        || name_str.ends_with(".png")
                        || name_str == "manifest.json"
                    {
                        let _ = fs::copy(&path, assets.join(&name));
                    }
                }
            }
            println!(
                "cargo:warning=Staged GPUPixel Android libs from {}",
                abi_dir.display()
            );
        } else {
            println!(
                "cargo:warning=GPUPixel Android libs missing at {} (beauty runtime optional until packaged)",
                abi_dir.display()
            );
        }
        return;
    }

    if matches!(target_os, "windows" | "macos") {
        if desktop_src.is_dir() {
            println!(
                "cargo:warning=GPUPixel desktop runtime present at {}",
                desktop_src.display()
            );
        } else {
            println!(
                "cargo:warning=GPUPixel desktop runtime missing at {} — run tools/gpupixel-ref/scripts/package-rr.ps1 after build",
                desktop_src.display()
            );
            let _ = fs::create_dir_all(&desktop_src);
            let placeholder = desktop_src.join("README.txt");
            if !placeholder.exists() {
                let _ = fs::write(
                    placeholder,
                    "Place libgpupixel_rr (+ deps, VNN model, lookup PNGs) here.\n\
                     See tools/gpupixel-ref/scripts/package-rr.ps1\n",
                );
            }
        }
    }
}

fn verify_sha256(path: &Path, expected_hash: &str) -> Result<bool, io::Error> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let hash_bytes = hasher.finalize();
    let calculated_hash = hex::encode(hash_bytes);
    Ok(calculated_hash == expected_hash)
}

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let (lib_name, expected_hash) =
        match (target_os.as_str(), target_arch.as_str()) {
            ("windows", "x86_64") => (
                "onnxruntime.dll",
                "579b636403983254346a5c1d80bd28f1519cd1e284cd204f8d4ff41f8d711559",
            ),
            ("windows", "aarch64") => (
                "onnxruntime.dll",
                "79281671a386ed1baab9dbdbb09fe55f99577011472e9526cf9d0b468bb6bcc7",
            ),
            ("linux", "x86_64") => (
                "libonnxruntime.so",
                "3da6146e14e7b8aaec625dde11d6114c7457c87a5f93d744897da8781e35c673",
            ),
            ("linux", "aarch64") => (
                "libonnxruntime.so",
                "0afd69a0ae38c5099fd0e8604dda398ac43dee67cd9c6394b5142b19e82528de",
            ),
            ("macos", "x86_64") => (
                "libonnxruntime.dylib",
                "283e595e61cf65df7a6b1d59a1616cbd35c8b6399dd90d799d99b71a3ff83160",
            ),
            ("macos", "aarch64") => (
                "libonnxruntime.dylib",
                "2b885992d3d6fa4130d39ec84a80d7504ff52750027c547bb22c86165f19406a",
            ),
            ("android", "aarch64") => (
                "libonnxruntime.so",
                "999ecfdb5b5a13e4097487773b6d71ce8a075408a237daab072e8f5e817bd78e",
            ),
            _ => panic!("Unsupported target: {}-{}", target_os, target_arch),
        };

    let dest_dir = if target_os == "android" {
        manifest_dir.join("libs").join("arm64-v8a")
    } else {
        manifest_dir.join("resources")
    };

    fs::create_dir_all(&dest_dir).unwrap();
    let dest_path = dest_dir.join(lib_name);

    if dest_path.exists() {
        match verify_sha256(&dest_path, expected_hash) {
            Ok(true) => {
                println!(
                    "cargo:warning=ONNX Runtime library already exists and is valid at {:?}.",
                    dest_path
                );
            }
            Ok(false) => {
                panic!(
                    "ONNX Runtime at {:?} has incorrect hash. Replace it with a verified local copy (remote download disabled).",
                    dest_path
                );
            }
            Err(e) => {
                panic!(
                    "Could not verify ONNX Runtime at {:?}: {}. Fix the local library (remote download disabled).",
                    dest_path, e
                );
            }
        }
    } else {
        panic!(
            "ONNX Runtime library missing at {:?}. Place {} for {}-{} under that path (remote download disabled).",
            dest_path, lib_name, target_os, target_arch
        );
    }

    if target_os == "android" {
        let jni_libs_dir = manifest_dir.join("gen/android/app/src/main/jniLibs/arm64-v8a");
        fs::create_dir_all(&jni_libs_dir).unwrap();
        fs::copy(&dest_path, jni_libs_dir.join(lib_name)).unwrap();

        println!("cargo:rustc-env=ORT_LIB_LOCATION={}", dest_dir.display());
        println!("cargo:rustc-env=ORT_STRATEGY=manual");
        println!("cargo:rustc-link-search=native={}", dest_dir.display());
    }

    // GPUPixel runtime: copy prebuilt tree from resources/gpupixel (desktop) or
    // libs/arm64-v8a (Android). Missing tree is OK at compile time; runtime reports
    // a clear error when beauty params are active.
    stage_gpupixel_runtime(&manifest_dir, &target_os);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=resources/gpupixel");
    // tauri-build does not watch the icons dir by default; without this,
    // changing icons/icon.ico does not re-link the exe and the old icon stays embedded.
    println!("cargo:rerun-if-changed=icons");

    // tauri-winres embeds the Common-Controls v6 manifest only into bin targets
    // (cargo:rustc-link-arg-bins), so `cargo test` binaries get no SxS
    // declaration. That only bites once a test binary's link closure imports
    // comctl32!TaskDialogIndirect (import-table parsing of the stock unit-test
    // binary shows no comctl32; the portrait-reshape test modules pull the
    // reference in): the loader then binds comctl32 5.82 and dies with
    // STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139) before any test runs. There is
    // no stable cargo directive that links a resource into lib unit-test
    // binaries only, so instead we drop the tauri-winres manifest entirely and
    // let the linker embed the identical manifest into every target
    // (bin / cdylib / tests) via /MANIFEST:EMBED; two embedded copies would
    // fail with CVT1100 (duplicate MANIFEST resource).
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let embed_manifest_via_linker = target_os == "windows" && target_env == "msvc";
    if embed_manifest_via_linker {
        let manifest_path = manifest_dir.join("tests-manifest.xml");
        println!("cargo:rerun-if-changed=tests-manifest.xml");
        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg=/MANIFESTINPUT:{}",
            manifest_path.display()
        );
    }

    let attributes = if embed_manifest_via_linker {
        tauri_build::Attributes::new()
            .windows_attributes(tauri_build::WindowsAttributes::new_without_app_manifest())
    } else {
        tauri_build::Attributes::default()
    };
    tauri_build::try_build(attributes).expect("failed to run tauri-build");
}
