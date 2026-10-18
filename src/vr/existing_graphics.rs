//! Attach a headset to the live Vulkan renderer; never migrate the world.

use super::{vk, VrBootstrap, VrResult, Vulkan, VulkanBinding};
use ash::vk::Handle;
use std::ffi::CStr;

/// Keep the extension entry points used by older OpenXR runtimes available,
/// including the KHR aliases of functions promoted into Vulkan 1.1.
pub(crate) fn create_desktop_instance(
    descriptor: &wgpu::InstanceDescriptor,
) -> VrResult<wgpu::Instance> {
    // SAFETY: enable only extensions advertised by the loader and retain every
    // setting required by wgpu. Ownership of the instance passes to wgpu once.
    unsafe {
        let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader: {e}"))?;
        let available = entry
            .enumerate_instance_extension_properties(None)
            .map_err(|e| format!("Vulkan instance extensions: {e}"))?;
        let extensions: Vec<_> = [
            ash::khr::external_memory_capabilities::NAME,
            ash::khr::external_fence_capabilities::NAME,
            ash::khr::external_semaphore_capabilities::NAME,
            ash::khr::get_physical_device_properties2::NAME,
        ]
        .into_iter()
        .filter(|name| {
            available
                .iter()
                .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == *name)
        })
        .collect();
        let hal = wgpu::hal::vulkan::Instance::init_with_callback(
            &wgpu::hal::InstanceDescriptor {
                name: "Bio-Spheres",
                flags: descriptor.flags,
                memory_budget_thresholds: descriptor.memory_budget_thresholds,
                backend_options: descriptor.backend_options.clone(),
            },
            Some(Box::new(move |args| {
                for extension in extensions {
                    if !args.extensions.contains(&extension) {
                        args.extensions.push(extension);
                    }
                }
            })),
        )
        .map_err(|e| format!("VR-compatible desktop instance: {e}"))?;
        Ok(wgpu::Instance::from_hal::<Vulkan>(hal))
    }
}

/// Prepare desktop Vulkan for a headset that may connect later. Enable the
/// sharing and memory-query extensions by name, including promoted KHR aliases.
pub(crate) fn request_desktop_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> VrResult<(wgpu::Device, wgpu::Queue)> {
    if !adapter.features().contains(descriptor.required_features)
        || !descriptor.required_limits.check_limits(&adapter.limits())
    {
        return Err("The requested desktop GPU features or limits are unsupported".into());
    }
    // SAFETY: use the adapter's own supported extensions/features and preserve
    // all wgpu-required device settings. wgpu owns the resulting device/queue.
    unsafe {
        let hal = adapter
            .as_hal::<Vulkan>()
            .ok_or("Desktop renderer is not Vulkan")?;
        let available = hal
            .shared_instance()
            .raw_instance()
            .enumerate_device_extension_properties(hal.raw_physical_device())
            .map_err(|e| format!("Vulkan device extensions: {e}"))?;
        let candidates: &[&'static CStr] = &[
            ash::khr::external_memory::NAME,
            ash::khr::external_fence::NAME,
            ash::khr::external_semaphore::NAME,
            ash::khr::timeline_semaphore::NAME,
            ash::khr::get_memory_requirements2::NAME,
            ash::khr::dedicated_allocation::NAME,
            ash::khr::bind_memory2::NAME,
            #[cfg(target_os = "windows")]
            ash::khr::win32_keyed_mutex::NAME,
            #[cfg(target_os = "windows")]
            ash::khr::external_memory_win32::NAME,
            #[cfg(target_os = "windows")]
            ash::khr::external_fence_win32::NAME,
            #[cfg(target_os = "windows")]
            ash::khr::external_semaphore_win32::NAME,
            #[cfg(unix)]
            ash::khr::external_memory_fd::NAME,
            #[cfg(unix)]
            ash::khr::external_fence_fd::NAME,
            #[cfg(unix)]
            ash::khr::external_semaphore_fd::NAME,
        ];
        let extensions: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|name| {
                available
                    .iter()
                    .any(|extension| CStr::from_ptr(extension.extension_name.as_ptr()) == *name)
            })
            .collect();
        let open = hal
            .open_with_callback(
                descriptor.required_features,
                &descriptor.memory_hints,
                Some(Box::new(move |args| {
                    for extension in extensions {
                        if !args.extensions.contains(&extension) {
                            args.extensions.push(extension);
                        }
                    }
                })),
            )
            .map_err(|e| format!("VR-compatible desktop device: {e}"))?;
        drop(hal);
        adapter
            .create_device_from_hal::<Vulkan>(open, descriptor)
            .map_err(|e| format!("Desktop wgpu device: {e}"))
    }
}

fn missing_extensions(required: &str, enabled: &[&CStr]) -> Vec<String> {
    // Older runtimes call the KHR entry points even when the functionality was
    // promoted into core Vulkan. Enable the named extensions explicitly so
    // vkGetDeviceProcAddr can return those aliases instead of null pointers.
    required
        .split_whitespace()
        .filter(|name| {
            !enabled
                .iter()
                .any(|enabled| enabled.to_bytes() == name.as_bytes())
        })
        .map(str::to_owned)
        .collect()
}

fn desktop_instance_api_version(loader_version: u32) -> u32 {
    // wgpu-hal 27 records the loader's supported version in InstanceShared,
    // but requests Vulkan 1.3 in VkApplicationInfo. A Vulkan 1.4 driver does
    // not turn this into a 1.4 instance (or make it incompatible with OpenXR).
    loader_version.min(vk::API_VERSION_1_3)
}

fn check_vulkan_version(
    api_version: u32,
    requirements: &openxr::vulkan::Requirements,
) -> VrResult<openxr::Version> {
    let version = openxr::Version::new(
        vk::api_version_major(api_version) as u16,
        vk::api_version_minor(api_version) as u16,
        0,
    );
    if version < requirements.min_api_version_supported {
        return Err(format!(
            "Desktop Vulkan {version} is below the headset's required version {}; scene untouched",
            requirements.min_api_version_supported,
        ));
    }
    // OpenXR defines maxApiVersionSupported as the highest version tested by
    // the runtime; newer compatible versions may work. SteamVR reports 1.2
    // here even on drivers supporting our 1.3 renderer. Let xrCreateSession
    // validate compatibility after checking extensions and GPU identity.
    Ok(version)
}

pub(super) fn validate_binding(
    bootstrap: &VrBootstrap,
    device: &wgpu::Device,
) -> VrResult<VulkanBinding> {
    if bootstrap.instance.exts().khr_vulkan_enable.is_none()
        || bootstrap.instance.exts().khr_vulkan_enable2.is_some()
    {
        return Err("Attaching a live device requires the original OpenXR Vulkan extension".into());
    }
    let requirements = bootstrap
        .instance
        .graphics_requirements::<openxr::Vulkan>(bootstrap.system)
        .map_err(|e| format!("OpenXR graphics requirements: {e}"))?;
    // SAFETY: borrow handles owned by this live device. No ownership transfer or
    // Vulkan queue operation occurs here; VrState retains clones of device/queue.
    unsafe {
        let hal = device.as_hal::<Vulkan>().ok_or(
            "The running renderer is not Vulkan; headset connection left the scene untouched",
        )?;
        let instance = hal.shared_instance();
        let api_version = desktop_instance_api_version(instance.instance_api_version());
        let xr_version = check_vulkan_version(api_version, &requirements)?;
        log::info!("Attaching headset to live Vulkan {xr_version} device (runtime minimum {}, highest tested {})",
            requirements.min_api_version_supported, requirements.max_api_version_supported);
        let required_instance = bootstrap
            .instance
            .vulkan_legacy_instance_extensions(bootstrap.system)
            .map_err(|e| format!("OpenXR instance extensions: {e}"))?;
        let required_device = bootstrap
            .instance
            .vulkan_legacy_device_extensions(bootstrap.system)
            .map_err(|e| format!("OpenXR device extensions: {e}"))?;
        let mut missing = missing_extensions(&required_instance, instance.extensions());
        missing.extend(missing_extensions(
            &required_device,
            hal.enabled_device_extensions(),
        ));
        if !missing.is_empty() {
            return Err(format!(
                "Headset requires missing Vulkan extensions {}; scene untouched",
                missing.join(", ")
            ));
        }
        let raw_instance = instance.raw_instance().handle().as_raw();
        let physical_device = bootstrap
            .instance
            .vulkan_graphics_device(bootstrap.system, raw_instance as _)
            .map_err(|e| format!("OpenXR graphics device: {e}"))?
            as u64;
        if physical_device != hal.raw_physical_device().as_raw() {
            return Err("The headset uses a different GPU; scene untouched".into());
        }
        log::info!("Validated live headset device and Vulkan sharing extensions");
        Ok(VulkanBinding {
            instance: raw_instance,
            physical_device,
            device: hal.raw_device().handle().as_raw(),
            queue_family_index: hal.queue_family_index(),
            queue_index: hal.queue_index(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_loader_does_not_change_the_desktop_instances_requested_version() {
        assert_eq!(
            desktop_instance_api_version(vk::make_api_version(0, 1, 4, 0)),
            vk::API_VERSION_1_3
        );
        assert_eq!(
            desktop_instance_api_version(vk::API_VERSION_1_3),
            vk::API_VERSION_1_3
        );
        assert_eq!(
            desktop_instance_api_version(vk::API_VERSION_1_2),
            vk::API_VERSION_1_2
        );
    }

    #[test]
    fn runtimes_highest_tested_version_does_not_block_live_attachment() {
        let requirements = openxr::vulkan::Requirements {
            min_api_version_supported: openxr::Version::new(1, 1, 0),
            max_api_version_supported: openxr::Version::new(1, 2, 0),
        };
        assert_eq!(
            check_vulkan_version(vk::API_VERSION_1_3, &requirements).unwrap(),
            openxr::Version::new(1, 3, 0)
        );
        assert!(check_vulkan_version(vk::API_VERSION_1_0, &requirements).is_err());
    }

    #[test]
    fn core_promotions_do_not_replace_runtime_requested_khr_entry_points() {
        let required = "VK_KHR_get_memory_requirements2 VK_KHR_win32_keyed_mutex";
        assert_eq!(
            missing_extensions(required, &[]),
            vec![
                "VK_KHR_get_memory_requirements2",
                "VK_KHR_win32_keyed_mutex"
            ]
        );
        assert_eq!(
            missing_extensions(required, &[ash::khr::get_memory_requirements2::NAME]),
            vec!["VK_KHR_win32_keyed_mutex"]
        );
        assert!(missing_extensions(
            required,
            &[
                ash::khr::get_memory_requirements2::NAME,
                ash::khr::win32_keyed_mutex::NAME
            ]
        )
        .is_empty());
    }

    #[test]
    fn desktop_device_supports_late_headset_interop_and_retains_live_buffers() {
        check_live_device(false);
    }

    #[test]
    #[ignore = "requires a connected OpenXR headset"]
    fn headset_attachment_preserves_existing_device_and_live_buffers() {
        check_live_device(true);
    }

    fn check_live_device(attach_headset: bool) {
        use wgpu::util::DeviceExt;
        if attach_headset {
            let _ = env_logger::Builder::new()
                .filter_module("bio_spheres::vr", log::LevelFilter::Info)
                .is_test(true)
                .try_init();
        }
        let instance = create_desktop_instance(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        })
        .unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let ray_queries = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let descriptor = wgpu::DeviceDescriptor {
            required_features: if ray_queries {
                wgpu::Features::EXPERIMENTAL_RAY_QUERY
            } else {
                wgpu::Features::empty()
            },
            required_limits: if ray_queries {
                wgpu::Limits::default().using_acceleration_structure_values(adapter.limits())
            } else {
                wgpu::Limits::default()
            },
            experimental_features: if ray_queries {
                // SAFETY: exercise the same opt-in as the desktop renderer.
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                Default::default()
            },
            ..Default::default()
        };
        let (device, queue) = request_desktop_device(&adapter, &descriptor).unwrap();
        assert!(device.features().contains(descriptor.required_features));
        // SteamVR calls these KHR aliases directly. Core Vulkan support alone
        // previously passed validation but left a null pointer in the runtime.
        unsafe {
            let hal = device.as_hal::<Vulkan>().unwrap();
            for (extension, function) in [
                (
                    ash::khr::get_memory_requirements2::NAME,
                    c"vkGetBufferMemoryRequirements2KHR",
                ),
                (
                    ash::khr::get_memory_requirements2::NAME,
                    c"vkGetImageMemoryRequirements2KHR",
                ),
                (ash::khr::bind_memory2::NAME, c"vkBindBufferMemory2KHR"),
                (ash::khr::timeline_semaphore::NAME, c"vkWaitSemaphoresKHR"),
            ] {
                if hal.enabled_device_extensions().contains(&extension) {
                    assert!(
                        hal.shared_instance()
                            .raw_instance()
                            .get_device_proc_addr(hal.raw_device().handle(), function.as_ptr())
                            .is_some(),
                        "Missing runtime Vulkan entry point {function:?}"
                    );
                }
            }
        }
        let live = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Live world state before headset connection"),
            contents: bytemuck::cast_slice(&[7u32, 31, 99, 1234]),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let before = unsafe { device.as_hal::<Vulkan>().unwrap().raw_device().handle() };
        let mut vr = attach_headset.then(|| {
            let bootstrap = VrBootstrap::for_existing_device().unwrap();
            super::super::VrState::on_existing_device(
                bootstrap,
                &device,
                &queue,
                wgpu::TextureFormat::Bgra8UnormSrgb,
                64,
                64,
                Default::default(),
            )
            .unwrap()
        });
        let gpu_access = vr.as_ref().map(|vr| vr.gpu_access()).unwrap_or_default();
        let _gpu = gpu_access.lock();
        if let Some(vr) = &mut vr {
            vr.set_immersive(true);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !vr.running() && std::time::Instant::now() < deadline {
                vr.begin_frame().unwrap();
                vr.end_frame().unwrap();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            assert!(
                vr.running(),
                "Connected headset session never reached READY"
            );
        }
        let xr_device = vr
            .as_ref()
            .map(|vr| vr.device.clone())
            .unwrap_or_else(|| device.clone());
        let xr_queue = vr
            .as_ref()
            .map(|vr| vr.queue.clone())
            .unwrap_or_else(|| queue.clone());
        if let Some(vr) = &vr {
            assert_eq!(vr.format(), wgpu::TextureFormat::Bgra8UnormSrgb);
        }
        assert_eq!(before, unsafe {
            xr_device.as_hal::<Vulkan>().unwrap().raw_device().handle()
        });
        let readback = xr_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = xr_device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&live, 0, &readback, 0, 16);
        xr_queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            send.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receive.recv().unwrap().unwrap();
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&readback.get_mapped_range(..)),
            &[7, 31, 99, 1234]
        );
        #[cfg(target_os = "windows")]
        unsafe {
            let hal = device.as_hal::<Vulkan>().unwrap();
            let available = hal
                .shared_instance()
                .raw_instance()
                .enumerate_device_extension_properties(hal.raw_physical_device())
                .unwrap();
            for name in [
                ash::khr::external_memory_win32::NAME,
                ash::khr::external_fence_win32::NAME,
                ash::khr::external_semaphore_win32::NAME,
                ash::khr::win32_keyed_mutex::NAME,
                ash::khr::timeline_semaphore::NAME,
            ] {
                if available
                    .iter()
                    .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == name)
                {
                    assert!(hal.enabled_device_extensions().contains(&name));
                }
            }
        }
        drop(vr);
    }
}
