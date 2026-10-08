// build.rs - Robust Adaptive CUDA Kernel Build Architecture for rust_localsleep_moe
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");

    println!("cargo:rerun-if-changed=csrc");

    // Only build CUDA if feature "cuda" is requested
    let cuda_feature = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_feature {
        println!("cargo:warning=[rust_localsleep_moe] Feature 'cuda' not enabled. Pure Rust engine mode.");
        return;
    }

    // Scan for any .cu CUDA kernel files
    let cu_files: Vec<PathBuf> = if csrc_dir.exists() {
        std::fs::read_dir(&csrc_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "cu"))
            .collect()
    } else {
        Vec::new()
    };

    if cu_files.is_empty() {
        println!("cargo:warning=[rust_localsleep_moe] No CUDA (.cu) files found in csrc. Skipping CUDA build.");
        return;
    }

    // Detect nvcc compiler
    let nvcc_found = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc_path = if nvcc_found {
        "nvcc".to_string()
    } else {
        let p = format!("{}/bin/nvcc", cuda_home);
        if Path::new(&p).exists() {
            p
        } else {
            println!("cargo:warning=[rust_localsleep_moe] nvcc not found in PATH or CUDA_HOME. Falling back to pure Rust.");
            return;
        }
    };

    println!("cargo:warning=[rust_localsleep_moe] Using nvcc at '{}' to compile {} CUDA kernels...", nvcc_path, cu_files.len());

    let mut build = cc::Build::new();
    build.cuda(true);
    build.compiler(&nvcc_path);
    build.flag("-cudart=shared");
    build.flag("-O3");
    build.flag("-std=c++17");

    // Check nvcc version to decide whether compute_120 is supported (requires CUDA >= 12.8)
    let nvcc_ver_out = Command::new(&nvcc_path).arg("--version").output();
    let mut support_sm120 = false;
    if let Ok(out) = nvcc_ver_out {
        let stdout = String::from_utf8_lossy(&out.stdout);
        if stdout.contains("release 12.8") || stdout.contains("release 12.9") || stdout.contains("release 13.") {
            support_sm120 = true;
        }
    }

    if support_sm120 {
        build.flag("-gencode=arch=compute_120,code=sm_120"); // Blackwell RTX 5080
    }
    // Universal fallback architectures: Ada (sm_89), Thor / Orin (sm_87)
    build.flag("-gencode=arch=compute_89,code=sm_89");
    build.flag("-gencode=arch=compute_87,code=sm_87");

    for cu_file in cu_files {
        build.file(cu_file);
    }

    build.include(&csrc_dir);
    build.compile("localsleep_cuda");

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
}
