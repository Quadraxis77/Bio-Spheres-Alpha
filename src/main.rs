#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

fn main() {
    #[cfg(feature = "vr")]
    if std::env::args().any(|arg| arg == "--vr-info" || arg == "--vr-check") {
        let result = if std::env::args().any(|arg| arg == "--vr-check") {
            bio_spheres::vr::check_graphics()
        } else {
            bio_spheres::vr::runtime_info()
        };
        match result {
            Ok(info) => println!("{info}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        return;
    }
    #[cfg(not(feature = "vr"))]
    if std::env::args().any(|arg| arg == "--vr" || arg == "--vr-info" || arg == "--vr-check") {
        eprintln!("Native VR requires a build with the vr feature enabled.");
        std::process::exit(1);
    }
    bio_spheres::app::run();
}
