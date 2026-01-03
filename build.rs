fn main() {
    // Embed default layout files
    println!("cargo:rerun-if-changed=default_dock_state_preview.ron");
    println!("cargo:rerun-if-changed=default_dock_state_gpu.ron");
    // Embed default UI state file
    println!("cargo:rerun-if-changed=default_ui_state.ron");
    
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let mut res = winres::WindowsResource::new();
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let icon_path = std::path::Path::new(&manifest_dir).join("assets").join("icon.ico");
        
        if icon_path.exists() {
            res.set_icon(icon_path.to_str().unwrap());
        }
        
        res.compile().unwrap();
    }
}
