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
        let mut files: Vec<PathBuf> = std::fs::read_dir(&csrc_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "cu"))
            .collect();
        files.sort();
        files
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

    println!(
        "cargo:warning=[rust_localsleep_moe] Using nvcc at '{}' to compile {} CUDA kernels...",
        nvcc_path,
        cu_files.len()
    );

    // Check nvcc version to decide whether compute_120 is supported (requires CUDA >= 12.8)
    let nvcc_ver_out = Command::new(&nvcc_path).arg("--version").output();
    let mut support_sm120 = false;
    if let Ok(out) = nvcc_ver_out {
        let stdout = String::from_utf8_lossy(&out.stdout);
        if stdout.contains("release 12.8")
            || stdout.contains("release 12.9")
            || stdout.contains("release 13.")
        {
            support_sm120 = true;
        }
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut obj_files: Vec<PathBuf> = Vec::new();

    // Compile each .cu file directly with nvcc (avoids cc-rs passing raw GCC flags like -ffunction-sections to nvcc)
    for cu_file in &cu_files {
        let stem = cu_file.file_stem().unwrap().to_string_lossy();
        let obj_path = out_dir.join(format!("{}.o", stem));

        let mut cmd = Command::new(&nvcc_path);
        cmd.arg("-c")
            .arg(cu_file)
            .arg("-o")
            .arg(&obj_path)
            .arg("-O3")
            .arg("-std=c++17")
            .arg("-cudart=shared")
            .arg("-Xcompiler")
            .arg("-fPIC,-O3,-fno-threadsafe-statics,-fno-exceptions")
            .arg("-I")
            .arg(&csrc_dir);

        if support_sm120 {
            cmd.arg("-gencode=arch=compute_120,code=sm_120"); // Blackwell RTX 5080
        }
        // Universal fallback architectures: Ada (sm_89), Thor / Orin (sm_87)
        cmd.arg("-gencode=arch=compute_89,code=sm_89");
        cmd.arg("-gencode=arch=compute_87,code=sm_87");

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("Failed to execute nvcc on {:?}: {}", cu_file, e));
        if !status.success() {
            panic!("nvcc failed to compile {:?}", cu_file);
        }
        obj_files.push(obj_path);
    }

    // Archive compiled .o files into liblocalsleep_cuda.a using ar
    let lib_path = out_dir.join("liblocalsleep_cuda.a");
    let _ = std::fs::remove_file(&lib_path);
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("crus").arg(&lib_path);
    for obj in &obj_files {
        ar_cmd.arg(obj);
    }
    let ar_status = ar_cmd
        .status()
        .expect("Failed to execute 'ar' to archive CUDA objects");
    if !ar_status.success() {
        panic!("Failed to create static archive {:?}", lib_path);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=localsleep_cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
