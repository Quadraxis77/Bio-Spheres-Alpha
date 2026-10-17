fn main() {
    println!("cargo:rerun-if-env-changed=BIO_SPHERES_BUG_REPORT_URL");
    println!("cargo:rerun-if-changed=bug-report-receiver/endpoint.txt");
    let report_url = std::env::var("BIO_SPHERES_BUG_REPORT_URL").unwrap_or_else(|_| {
        include_str!("bug-report-receiver/endpoint.txt")
            .trim()
            .to_owned()
    });
    println!(
        "cargo:rustc-env=BIO_SPHERES_BUG_REPORT_URL={}",
        report_url.trim()
    );
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    let mut sources = Vec::new();
    collect_sources(std::path::Path::new("src"), &mut sources);
    collect_sources(std::path::Path::new("shaders"), &mut sources);
    sources.extend(["Cargo.toml", "Cargo.lock", "build.rs"].map(std::path::PathBuf::from));
    sources.sort();
    let mut fingerprint = 0xcbf29ce484222325_u64;
    for path in sources {
        for byte in path
            .to_string_lossy()
            .replace('\\', "/")
            .bytes()
            .chain(std::fs::read(&path).unwrap_or_default())
        {
            fingerprint = (fingerprint ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    println!("cargo:rustc-env=BIO_SPHERES_SOURCE_ID={fingerprint:016x}");
    println!(
        "cargo:rustc-env=BIO_SPHERES_BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_default()
    );
    let built = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    println!("cargo:rustc-env=BIO_SPHERES_BUILT_AT={built}");
    // Embed default layout files
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=assets/icon.png");
    println!("cargo:rerun-if-changed=default_dock_state_preview.ron");
    println!("cargo:rerun-if-changed=default_dock_state_gpu.ron");
    // Embed default UI state file
    println!("cargo:rerun-if-changed=default_ui_state.ron");
    // Embed runtime settings files (extracted on first launch)
    println!("cargo:rerun-if-changed=cave_settings.ron");
    println!("cargo:rerun-if-changed=cell_visuals.ron");
    println!("cargo:rerun-if-changed=fluid_settings.ron");
    println!("cargo:rerun-if-changed=light_settings.ron");
    println!("cargo:rerun-if-changed=sun_settings.ron");
    println!("cargo:rerun-if-changed=fluid_render_settings.ron");

    // Rebuild if cave system shaders change
    println!("cargo:rerun-if-changed=shaders/cave_system.wgsl");
    println!("cargo:rerun-if-changed=shaders/cave_collision.wgsl");
    println!("cargo:rerun-if-changed=shaders/cave_spatial_grid_build.wgsl");

    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let mut res = winres::WindowsResource::new();
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let icon_path = std::path::Path::new(&manifest_dir)
            .join("assets")
            .join("icon.ico");

        if icon_path.exists() {
            res.set_icon(icon_path.to_str().unwrap());
        }

        res.compile().unwrap();
    }
}

fn collect_sources(directory: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_sources(&path, files);
            } else if path
                .extension()
                .is_some_and(|ext| ext == "rs" || ext == "wgsl")
            {
                files.push(path);
            }
        }
    }
}
