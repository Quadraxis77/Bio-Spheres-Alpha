//! Native OpenXR presentation using the same Vulkan device as the wgpu renderer.

use ash::vk::{self, Handle};
use glam::{Quat, Vec3};
use openxr as xr;
use std::sync::{Arc, Mutex};
mod controllers;
pub mod controls;
pub mod dial;
mod fade;
mod frame_wait;
pub mod input;
mod inspector;
mod locomotion;
mod mirror;
mod panel;
pub mod population;
mod runtime;
mod surface;
mod theme;
pub(crate) mod ui_readability;
pub use panel::PanelSettings;

pub type VrResult<T> = Result<T, String>;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const UI_SUPERSAMPLE_FACTOR: f32 = 2.0;

fn supersampled_dimensions(width: u32, height: u32, max_dimension: u32) -> (u32, u32, f32) {
    let width = width.max(1);
    let height = height.max(1);
    let max_dimension = max_dimension.max(1);
    let scale = UI_SUPERSAMPLE_FACTOR
        .min(max_dimension as f32 / width as f32)
        .min(max_dimension as f32 / height as f32);
    let scaled_width = ((width as f32 * scale).round() as u32).clamp(1, max_dimension);
    let scaled_height = ((height as f32 * scale).round() as u32).clamp(1, max_dimension);
    let actual_scale =
        (scaled_width as f32 / width as f32).min(scaled_height as f32 / height as f32);
    (scaled_width, scaled_height, actual_scale)
}
const MINIMUM_VULKAN_API_VERSION: u32 = vk::API_VERSION_1_2;
const MAXIMUM_VULKAN_API_VERSION: u32 = vk::API_VERSION_1_3;
type Vulkan = wgpu::hal::api::Vulkan;

fn vulkan_version(version: xr::Version) -> u32 {
    vk::make_api_version(
        0,
        version.major() as u32,
        version.minor() as u32,
        version.patch(),
    )
}

fn select_vulkan_api_version(
    minimum: xr::Version,
    maximum: xr::Version,
    loader_version: u32,
) -> VrResult<u32> {
    let minimum = vulkan_version(minimum);
    let maximum = vulkan_version(maximum);
    if maximum < MINIMUM_VULKAN_API_VERSION {
        return Err(format!(
            "Native VR requires Vulkan 1.2 for timeline semaphores; the active OpenXR runtime supports {} through {}.",
            xr::Version::new(
                vk::api_version_major(minimum) as u16,
                vk::api_version_minor(minimum) as u16,
                vk::api_version_patch(minimum),
            ),
            xr::Version::new(
                vk::api_version_major(maximum) as u16,
                vk::api_version_minor(maximum) as u16,
                vk::api_version_patch(maximum),
            ),
        ));
    }
    if loader_version < MINIMUM_VULKAN_API_VERSION {
        return Err(format!(
            "Native VR needs a Vulkan 1.2 loader for timeline semaphores; the loader supports Vulkan {}. Update the graphics driver.",
            xr::Version::new(
                vk::api_version_major(loader_version) as u16,
                vk::api_version_minor(loader_version) as u16,
                vk::api_version_patch(loader_version),
            ),
        ));
    }
    let selected = MAXIMUM_VULKAN_API_VERSION.min(maximum).min(loader_version);
    if selected < minimum {
        return Err(format!(
            "The active OpenXR runtime requires Vulkan {}, but this application and loader can provide at most Vulkan {}.",
            xr::Version::new(
                vk::api_version_major(minimum) as u16,
                vk::api_version_minor(minimum) as u16,
                vk::api_version_patch(minimum),
            ),
            xr::Version::new(
                vk::api_version_major(selected) as u16,
                vk::api_version_minor(selected) as u16,
                vk::api_version_patch(selected),
            ),
        ));
    }
    Ok(selected)
}

pub struct VrBootstrap {
    instance: xr::Instance,
    system: xr::SystemId,
    refresh_rate_supported: bool,
}

pub struct VrGraphics {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    binding: VulkanBinding,
}

// Vulkan handles are values owned by the thread-safe wgpu objects above, not
// pointers to Rust data. Reconstruct the FFI binding only at session creation.
struct VulkanBinding {
    instance: u64,
    physical_device: u64,
    device: u64,
    queue_family_index: u32,
    queue_index: u32,
}

impl VrBootstrap {
    pub fn new() -> VrResult<Self> {
        runtime::connect(Self::connect_runtime)
    }

    fn connect_runtime() -> VrResult<Self> {
        let (instance, refresh_rate_supported) = create_instance()?;
        let properties = instance
            .properties()
            .map_err(|e| format!("OpenXR runtime properties: {e}"))?;
        log::info!(
            "OpenXR runtime: {} {}",
            properties.runtime_name,
            properties.runtime_version
        );
        let system = instance
            .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
            .map_err(|e| {
                format!("OpenXR headset: {e}. Connect the headset and start its PCVR connection.")
            })?;
        Ok(Self {
            instance,
            system,
            refresh_rate_supported,
        })
    }
}

fn create_instance() -> VrResult<(xr::Instance, bool)> {
    let entry = xr::Entry::linked(&()).map_err(|e| format!("OpenXR loader: {e}"))?;
    let supported = entry
        .enumerate_extensions()
        .map_err(|e| format!("OpenXR extensions: {e}"))?;
    if !supported.khr_vulkan_enable2 {
        return Err(
            "The active OpenXR runtime does not support Vulkan graphics (XR_KHR_vulkan_enable2)."
                .into(),
        );
    }
    let mut extensions = xr::ExtensionSet::default();
    extensions.khr_vulkan_enable2 = true;
    extensions.fb_display_refresh_rate = supported.fb_display_refresh_rate;
    extensions.ext_user_presence = supported.ext_user_presence;
    extensions.meta_touch_controller_plus = supported.meta_touch_controller_plus;
    extensions.fb_touch_controller_pro = supported.fb_touch_controller_pro;
    let instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "Biospheres",
                application_version: 1,
                engine_name: "Biospheres",
                engine_version: 1,
                api_version: xr::Version::new(1, 0, 0),
            },
            &extensions,
            &[],
            &(),
        )
        .map_err(|e| format!("OpenXR instance: {e}"))?;
    Ok((instance, supported.fb_display_refresh_rate))
}

impl VrBootstrap {
    /// The runtime creates the Vulkan instance/device and chooses the physical GPU.
    /// wgpu then assumes ownership of those same handles; there is no second GPU device.
    pub fn create_graphics(&self) -> VrResult<VrGraphics> {
        let requirements = self
            .instance
            .graphics_requirements::<xr::Vulkan>(self.system)
            .map_err(|e| format!("OpenXR graphics requirements: {e}"))?;
        // SAFETY: all Vulkan handles are created through OpenXR, then handed to
        // wgpu-hal exactly once. ash clones are borrowed function tables, not owners.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader: {e}"))?;
            let loader_version = entry
                .try_enumerate_instance_version()
                .map_err(|e| format!("Vulkan loader version: {e}"))?
                .unwrap_or(vk::API_VERSION_1_0);
            let api_version = select_vulkan_api_version(
                requirements.min_api_version_supported,
                requirements.max_api_version_supported,
                loader_version,
            )?;
            let flags = wgpu::InstanceFlags::empty();
            let extensions =
                wgpu::hal::vulkan::Instance::desired_extensions(&entry, api_version, flags)
                    .map_err(|e| format!("Vulkan instance extensions: {e}"))?;
            let extension_ptrs: Vec<_> = extensions.iter().map(|name| name.as_ptr()).collect();
            let app_info = vk::ApplicationInfo::default().api_version(api_version);
            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extension_ptrs);
            let raw_instance = self
                .instance
                .create_vulkan_instance(
                    self.system,
                    std::mem::transmute(entry.static_fn().get_instance_proc_addr),
                    &create_info as *const _ as *const _,
                )
                .map_err(|e| format!("OpenXR Vulkan instance: {e}"))?
                .map_err(|e| format!("Vulkan instance: {:?}", vk::Result::from_raw(e)))?;
            let raw_instance =
                ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(raw_instance as _));
            let physical = vk::PhysicalDevice::from_raw(
                self.instance
                    .vulkan_graphics_device(self.system, raw_instance.handle().as_raw() as _)
                    .map_err(|e| format!("OpenXR graphics device: {e}"))? as _,
            );
            let family = raw_instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|p| {
                    p.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .ok_or("The headset GPU has no graphics/compute queue")?
                as u32;
            let instance_handle = raw_instance.handle();
            let hal_instance = wgpu::hal::vulkan::Instance::from_raw(
                entry,
                raw_instance,
                api_version,
                0,
                None,
                extensions,
                flags,
                Default::default(),
                false,
                None,
            )
            .map_err(|e| format!("wgpu Vulkan instance: {e}"))?;
            let exposed = hal_instance
                .expose_adapter(physical)
                .ok_or("wgpu cannot use the headset GPU")?;
            let instance = wgpu::Instance::from_hal::<Vulkan>(hal_instance);
            let adapter = instance.create_adapter_from_hal::<Vulkan>(exposed);
            let timestamp_features =
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            let features = if adapter.features().contains(timestamp_features) {
                timestamp_features
            } else {
                wgpu::Features::empty()
            };
            let limits = adapter.limits();
            let desc = wgpu::DeviceDescriptor {
                label: Some("Biospheres OpenXR Device"),
                required_features: features,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 64
                        .min(limits.max_storage_buffers_per_shader_stage),
                    max_storage_buffer_binding_size: limits
                        .max_storage_buffer_binding_size
                        .min(512 * 1024 * 1024),
                    max_buffer_size: limits.max_buffer_size.min(512 * 1024 * 1024),
                    ..Default::default()
                },
                ..Default::default()
            };
            let open_device;
            let device_handle;
            {
                let hal_adapter = adapter.as_hal::<Vulkan>().ok_or("Missing Vulkan adapter")?;
                let device_extensions = hal_adapter.required_device_extensions(features);
                let extension_ptrs: Vec<_> =
                    device_extensions.iter().map(|name| name.as_ptr()).collect();
                let mut physical_features =
                    hal_adapter.physical_device_features(&device_extensions, features);
                let priorities = [1.0];
                let queues = [vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)];
                let create_info = physical_features.add_to_device_create(
                    vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queues)
                        .enabled_extension_names(&extension_ptrs),
                );
                let shared = hal_adapter.shared_instance();
                let raw_device = self
                    .instance
                    .create_vulkan_device(
                        self.system,
                        std::mem::transmute(shared.entry().static_fn().get_instance_proc_addr),
                        physical.as_raw() as _,
                        &create_info as *const _ as *const _,
                    )
                    .map_err(|e| format!("OpenXR Vulkan device: {e}"))?
                    .map_err(|e| format!("Vulkan device: {:?}", vk::Result::from_raw(e)))?;
                let raw_device = ash::Device::load(
                    shared.raw_instance().fp_v1_0(),
                    vk::Device::from_raw(raw_device as _),
                );
                // Check the promoted entry point before giving the device to wgpu.
                // Ash's null-pointer stub panics across an extern boundary otherwise.
                let wait_semaphores =
                    if device_extensions.contains(&ash::khr::timeline_semaphore::NAME) {
                        c"vkWaitSemaphoresKHR"
                    } else {
                        c"vkWaitSemaphores"
                    };
                if shared
                    .raw_instance()
                    .get_device_proc_addr(raw_device.handle(), wait_semaphores.as_ptr())
                    .is_none()
                {
                    raw_device.destroy_device(None);
                    return Err("The VR Vulkan device is missing vkWaitSemaphores. The driver/runtime did not provide timeline semaphore synchronization.".into());
                }
                device_handle = raw_device.handle();
                open_device = hal_adapter
                    .device_from_raw(
                        raw_device,
                        None,
                        &device_extensions,
                        features,
                        &desc.memory_hints,
                        family,
                        0,
                    )
                    .map_err(|e| format!("wgpu Vulkan device: {e}"))?;
            }
            let (device, queue) = adapter
                .create_device_from_hal::<Vulkan>(open_device, &desc)
                .map_err(|e| format!("wgpu device: {e}"))?;
            Ok(VrGraphics {
                instance,
                adapter,
                device,
                queue,
                binding: VulkanBinding {
                    instance: instance_handle.as_raw(),
                    physical_device: physical.as_raw(),
                    device: device_handle.as_raw(),
                    queue_family_index: family,
                    queue_index: 0,
                },
            })
        }
    }
}

struct RuntimeSwapchain {
    // Texture callbacks retain a swapchain clone until wgpu releases the image.
    textures: Vec<wgpu::Texture>,
    swapchain: Arc<Mutex<xr::Swapchain<xr::Vulkan>>>,
    acquired: Option<u32>,
    width: u32,
    height: u32,
}

impl RuntimeSwapchain {
    fn new(
        session: &xr::Session<xr::Vulkan>,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        layers: u32,
    ) -> VrResult<Self> {
        let vk_format = match format {
            wgpu::TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
            wgpu::TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
            wgpu::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            wgpu::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            _ => return Err(format!("Unsupported OpenXR render format: {format:?}")),
        };
        let formats = session
            .enumerate_swapchain_formats()
            .map_err(|e| format!("OpenXR formats: {e}"))?;
        if !formats.contains(&(vk_format.as_raw() as _)) {
            return Err(format!(
                "The runtime does not support the desktop render format {format:?}"
            ));
        }
        let swapchain = session
            .create_swapchain(&xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                    | xr::SwapchainUsageFlags::SAMPLED,
                format: vk_format.as_raw() as _,
                sample_count: 1,
                width,
                height,
                face_count: 1,
                array_size: layers,
                mip_count: 1,
            })
            .map_err(|e| format!("OpenXR swapchain: {e}"))?;
        let images = swapchain
            .enumerate_images()
            .map_err(|e| format!("OpenXR images: {e}"))?;
        let swapchain = Arc::new(Mutex::new(swapchain));
        let descriptor = wgpu::TextureDescriptor {
            label: Some("OpenXR Runtime Image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let mut textures = Vec::new();
        for image in images {
            // SAFETY: image belongs to this session's Vulkan device. The callback
            // prevents wgpu from destroying runtime memory and retains its owner.
            let texture = unsafe {
                let hal_device = device.as_hal::<Vulkan>().ok_or("Missing Vulkan device")?;
                let owner = swapchain.clone();
                let hal_texture = hal_device.texture_from_raw(
                    vk::Image::from_raw(image as _),
                    &wgpu::hal::TextureDescriptor {
                        label: descriptor.label,
                        size: descriptor.size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: descriptor.dimension,
                        format,
                        usage: wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::RESOURCE,
                        memory_flags: wgpu::hal::MemoryFlags::empty(),
                        view_formats: vec![],
                    },
                    Some(Box::new(move || drop(owner))),
                );
                drop(hal_device);
                device.create_texture_from_hal::<Vulkan>(hal_texture, &descriptor)
            };
            textures.push(texture);
        }
        Ok(Self {
            textures,
            swapchain,
            acquired: None,
            width,
            height,
        })
    }

    fn acquire(&mut self) -> VrResult<()> {
        let mut swapchain = self.swapchain.lock().unwrap();
        self.acquired = Some(
            swapchain
                .acquire_image()
                .map_err(|e| format!("OpenXR acquire image: {e}"))?,
        );
        swapchain
            .wait_image(xr::Duration::INFINITE)
            .map_err(|e| format!("OpenXR wait image: {e}"))
    }

    fn view(&self, layer: u32) -> Option<wgpu::TextureView> {
        Some(
            self.textures[self.acquired? as usize].create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            }),
        )
    }

    fn release(&mut self) -> VrResult<()> {
        if self.acquired.take().is_some() {
            self.swapchain
                .lock()
                .unwrap()
                .release_image()
                .map_err(|e| format!("OpenXR release image: {e}"))?;
        }
        Ok(())
    }

    fn rect(&self) -> xr::Rect2Di {
        xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: self.width as i32,
                height: self.height as i32,
            },
        }
    }
}

pub struct VrState {
    actions: input::Actions,
    mirror: mirror::Mirror,
    fade: fade::Fade,
    controller_renderer: controllers::Controllers,
    hands: RuntimeSwapchain,
    hands_drawn: bool,
    tracked_input: std::cell::RefCell<input::VrInput>,
    eyes: RuntimeSwapchain,
    ui: RuntimeSwapchain,
    ui_pixel_scale: f32,
    wheel: RuntimeSwapchain,
    wheel_pixel_scale: f32,
    wheel_ctx: egui::Context,
    wheel_renderer: egui_wgpu::Renderer,
    wheel_pose: Option<xr::Posef>,
    frame_stream: xr::FrameStream<xr::Vulkan>,
    frame_waiter: frame_wait::FrameWait,
    session_epoch: u64,
    local: xr::Space,
    view_space: xr::Space,
    session: xr::Session<xr::Vulkan>,
    instance: xr::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    running: bool,
    focused: bool,
    visible: bool,
    user_present: Option<bool>,
    display_refresh_hz: Option<f32>,
    exiting: bool,
    frame: Option<xr::FrameState>,
    views: Vec<xr::View>,
    ui_anchor: panel::PanelAnchor,
    screen_surface: surface::Surface,
    ui_projected: bool,
    ui_space_change: Option<xr::Time>,
    immersive: bool,
    pub world_units_per_meter: f32,
    eyes_drawn: bool,
    ui_drawn: bool,
    format: wgpu::TextureFormat,
    last_frame_report: Option<std::time::Instant>,
    last_input_report: std::cell::Cell<Option<std::time::Instant>>,
    submitted_frames: u64,
}

impl VrState {
    pub fn new(
        bootstrap: VrBootstrap,
        graphics: &VrGraphics,
        formats: &[wgpu::TextureFormat],
        ui_width: u32,
        ui_height: u32,
    ) -> VrResult<Self> {
        // SAFETY: binding names the exact instance/device/queue owned by graphics.
        let (session, frame_waiter, frame_stream) = unsafe {
            bootstrap.instance.create_session::<xr::Vulkan>(
                bootstrap.system,
                &xr::vulkan::SessionCreateInfo {
                    instance: graphics.binding.instance as _,
                    physical_device: graphics.binding.physical_device as _,
                    device: graphics.binding.device as _,
                    queue_family_index: graphics.binding.queue_family_index,
                    queue_index: graphics.binding.queue_index,
                },
            )
        }
        .map_err(|e| format!("OpenXR session: {e}"))?;
        let runtime_formats = session
            .enumerate_swapchain_formats()
            .map_err(|e| format!("OpenXR formats: {e}"))?;
        let format = [
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .into_iter()
        .find(|format| {
            let native = match format {
                wgpu::TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
                wgpu::TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
                wgpu::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
                _ => vk::Format::R8G8B8A8_UNORM,
            };
            formats.contains(format) && runtime_formats.contains(&(native.as_raw() as _))
        })
        .ok_or("No color format is shared by the OpenXR runtime and desktop surface")?;
        let views = bootstrap
            .instance
            .enumerate_view_configuration_views(bootstrap.system, VIEW_TYPE)
            .map_err(|e| format!("OpenXR stereo views: {e}"))?;
        if views.len() != 2 {
            return Err("The OpenXR runtime did not provide two stereo views".into());
        }
        let width = views
            .iter()
            .map(|v| v.recommended_image_rect_width)
            .max()
            .unwrap();
        let height = views
            .iter()
            .map(|v| v.recommended_image_rect_height)
            .max()
            .unwrap();
        let eyes = RuntimeSwapchain::new(&session, &graphics.device, format, width, height, 2)?;
        let (ui_width, ui_height, ui_pixel_scale) = supersampled_dimensions(
            ui_width,
            ui_height,
            graphics.device.limits().max_texture_dimension_2d,
        );
        log::info!("OpenXR UI resolution: {ui_width} x {ui_height} ({ui_pixel_scale:.2}x)");
        let ui = RuntimeSwapchain::new(&session, &graphics.device, format, ui_width, ui_height, 2)?;
        let (wheel_width, wheel_height, wheel_pixel_scale) = supersampled_dimensions(
            controls::WHEEL_PIXELS,
            controls::WHEEL_PIXELS,
            graphics.device.limits().max_texture_dimension_2d,
        );
        let local = session
            .create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)
            .map_err(|e| format!("OpenXR local space: {e}"))?;
        let view_space = session
            .create_reference_space(xr::ReferenceSpaceType::VIEW, xr::Posef::IDENTITY)
            .map_err(|e| format!("OpenXR head space: {e}"))?;
        let actions = input::Actions::new(&bootstrap.instance, &session)?;
        if bootstrap.refresh_rate_supported {
            if let Ok(rates) = session.enumerate_display_refresh_rates() {
                if rates.iter().any(|rate| (*rate - 120.0).abs() < 0.1) {
                    if let Err(e) = session.request_display_refresh_rate(120.0) {
                        log::warn!(
                            "OpenXR 120 Hz request: {e}; retaining the runtime's current rate"
                        );
                    }
                } else {
                    log::warn!("OpenXR exposes {rates:?} Hz, so this session cannot run at 120 FPS. Set the PCVR runtime's headset refresh rate to 120 Hz before connecting.");
                }
                log::info!("OpenXR refresh rates: {rates:?}");
            }
        }
        let display_refresh_hz = if bootstrap.refresh_rate_supported {
            session.get_display_refresh_rate().ok()
        } else {
            None
        };
        log::info!("OpenXR eye resolution: {width} x {height}");
        Ok(Self {
            wheel: RuntimeSwapchain::new(
                &session,
                &graphics.device,
                format,
                wheel_width,
                wheel_height,
                1,
            )?,
            wheel_pixel_scale,
            wheel_ctx: egui::Context::default(),
            wheel_renderer: egui_wgpu::Renderer::new(&graphics.device, format, Default::default()),
            wheel_pose: None,
            actions,
            mirror: mirror::Mirror::new(&graphics.device, format),
            fade: fade::Fade::new(&graphics.device, format),
            controller_renderer: controllers::Controllers::new(
                &graphics.device,
                &graphics.queue,
                format,
                (width / 2).max(1),
                (height / 2).max(1),
            ),
            hands: RuntimeSwapchain::new(
                &session,
                &graphics.device,
                format,
                (width / 2).max(1),
                (height / 2).max(1),
                2,
            )?,
            hands_drawn: false,
            tracked_input: std::cell::RefCell::new(Default::default()),
            eyes,
            ui,
            ui_pixel_scale,
            frame_stream,
            frame_waiter: frame_wait::FrameWait::new(
                frame_waiter,
                graphics.queue.clone(),
                graphics.device.clone(),
            ),
            session_epoch: 0,
            local,
            view_space,
            session,
            instance: bootstrap.instance,
            device: graphics.device.clone(),
            queue: graphics.queue.clone(),
            running: false,
            focused: false,
            visible: false,
            user_present: None,
            display_refresh_hz,
            exiting: false,
            frame: None,
            views: vec![],
            ui_anchor: panel::PanelAnchor::default(),
            screen_surface: surface::Surface::new(&graphics.device, format),
            ui_projected: false,
            ui_space_change: None,
            immersive: false,
            world_units_per_meter: 20.0,
            eyes_drawn: false,
            ui_drawn: false,
            format,
            last_frame_report: None,
            last_input_report: std::cell::Cell::new(None),
            submitted_frames: 0,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.eyes.width, self.eyes.height)
    }
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
    pub fn ui_dimensions(&self) -> (u32, u32) {
        (self.ui.width, self.ui.height)
    }
    pub fn ui_pixel_scale(&self) -> f32 {
        self.ui_pixel_scale
    }
    pub fn running(&self) -> bool {
        self.running
    }
    pub fn headset_active(&self) -> bool {
        self.running && self.visible && self.user_present.unwrap_or(self.focused)
    }
    pub fn presenting(&self) -> bool {
        self.headset_active() && self.frame.is_some_and(|frame| frame.should_render)
    }
    pub fn display_refresh_hz(&self) -> Option<f32> {
        self.display_refresh_hz
    }
    pub fn frame_active(&self) -> bool {
        self.frame.is_some()
    }
    pub fn set_immersive(&mut self, immersive: bool) {
        if self.immersive && !immersive {
            // Returning to the editor/menu places its panel in front of the
            // player's current position, then leaves it there.
            self.ui_anchor.reset();
        }
        self.immersive = immersive;
    }
    pub fn should_exit(&self) -> bool {
        self.exiting
    }
    pub fn input(&self) -> VrResult<input::VrInput> {
        let Some(frame) = self.frame else {
            return Ok(Default::default());
        };
        let sampled = self.actions.sample(
            &self.session,
            &self.local,
            &self.view_space,
            frame.predicted_display_time,
            self.focused && self.headset_active() && frame.should_render,
            self.ui_anchor.pose(),
            self.ui.width,
            self.ui.height,
            self.ui_anchor.settings,
        );
        if sampled.is_err() {
            *self.tracked_input.borrow_mut() = Default::default();
        }
        let mut input = sampled?;
        if self.input_focused() && frame.should_render && self.views.len() == 2 {
            input.head_position.get_or_insert_with(|| {
                (input::position(self.views[0].pose) + input::position(self.views[1].pose)) * 0.5
            });
            input
                .head_rotation
                .get_or_insert_with(|| input::rotation(self.views[0].pose));
        }
        *self.tracked_input.borrow_mut() = input.clone();
        if self
            .last_input_report
            .get()
            .is_none_or(|at| at.elapsed().as_secs() >= 5)
        {
            log::info!(
                "OpenXR input: focused={}, aims={:?}, triggers={:?}, menus={:?}, panel_pointer={:?}, grips={:?}, wheel_button={}, stick_clicks={:?}, sticks={:?}",
                self.input_focused(), input.aims.map(|pose| pose.is_some()),
                input.triggers, input.menus, input.pointer, input.grips.map(|pose| pose.is_some()),
                input.wheel_button, input.stick_clicks, (input.movement, input.turn, input.lift),
            );
            self.last_input_report.set(Some(std::time::Instant::now()));
        }
        Ok(input)
    }
    pub fn input_focused(&self) -> bool {
        self.focused && self.headset_active()
    }
    pub fn resize_ui(&mut self, width: u32, height: u32) -> VrResult<()> {
        if self.frame_active() {
            return Err("Cannot resize an OpenXR swapchain during a frame".into());
        }
        let (width, height, pixel_scale) =
            supersampled_dimensions(width, height, self.device.limits().max_texture_dimension_2d);
        if (self.ui.width, self.ui.height) == (width, height) {
            self.ui_pixel_scale = pixel_scale;
            return Ok(());
        }
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| e.to_string())?;
        self.ui =
            RuntimeSwapchain::new(&self.session, &self.device, self.format, width, height, 2)?;
        self.ui_pixel_scale = pixel_scale;
        Ok(())
    }
    pub fn capture_left_eye(&self, queue: &wgpu::Queue, target: &wgpu::TextureView) {
        if let Some(source) = self.eyes.view(0) {
            self.mirror.render(&self.device, queue, &source, target);
        }
    }

    pub fn begin_frame(&mut self) -> VrResult<()> {
        let mut events = xr::EventDataBuffer::new();
        while let Some(event) = self
            .instance
            .poll_event(&mut events)
            .map_err(|e| format!("OpenXR events: {e}"))?
        {
            match event {
                xr::Event::SessionStateChanged(event) => {
                    log::info!("OpenXR session: {:?}", event.state());
                    self.focused = event.state() == xr::SessionState::FOCUSED;
                    self.visible = matches!(
                        event.state(),
                        xr::SessionState::VISIBLE | xr::SessionState::FOCUSED
                    );
                    match event.state() {
                        xr::SessionState::READY => {
                            self.session
                                .begin(VIEW_TYPE)
                                .map_err(|e| format!("OpenXR begin session: {e}"))?;
                            self.running = true;
                            self.ui_anchor.reset();
                            self.ui_space_change = None;
                        }
                        xr::SessionState::STOPPING => {
                            self.session
                                .end()
                                .map_err(|e| format!("OpenXR end session: {e}"))?;
                            self.running = false;
                            self.session_epoch = self.session_epoch.wrapping_add(1);
                            self.user_present = None;
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            self.exiting = true
                        }
                        _ => {}
                    }
                }
                xr::Event::InstanceLossPending(_) => self.exiting = true,
                xr::Event::UserPresenceChangedEXT(event) => {
                    self.user_present = Some(event.is_user_present());
                    log::info!("OpenXR headset worn: {}", event.is_user_present());
                }
                xr::Event::DisplayRefreshRateChangedFB(event) => {
                    self.display_refresh_hz = Some(event.to_display_refresh_rate());
                    log::info!(
                        "OpenXR display refresh rate: {} Hz",
                        event.to_display_refresh_rate()
                    );
                }
                xr::Event::ReferenceSpaceChangePending(event)
                    if event.reference_space_type() == xr::ReferenceSpaceType::LOCAL =>
                {
                    self.ui_space_change = Some(event.change_time());
                }
                _ => {}
            }
        }
        if !self.running || self.exiting {
            return Ok(());
        }
        let Some(frame) = self
            .frame_waiter
            .poll(self.session_epoch, self.headset_active())?
        else {
            return Ok(());
        };
        self.frame_stream
            .begin()
            .map_err(|e| format!("OpenXR begin frame: {e}"))?;
        self.frame = Some(frame);
        self.eyes_drawn = false;
        self.hands_drawn = false;
        self.ui_drawn = false;
        self.ui_projected = false;
        self.wheel_pose = None;
        self.views.clear();
        if self
            .ui_space_change
            .is_some_and(|time| frame.predicted_display_time.as_nanos() >= time.as_nanos())
        {
            self.ui_anchor.reset();
            self.ui_space_change = None;
        }
        if frame.should_render && self.headset_active() {
            if let Ok(head) = self
                .view_space
                .locate(&self.local, frame.predicted_display_time)
            {
                if head.location_flags.contains(
                    xr::SpaceLocationFlags::POSITION_VALID
                        | xr::SpaceLocationFlags::ORIENTATION_VALID,
                ) {
                    self.ui_anchor.capture(head.pose);
                }
            }
            let (validity, views) = self
                .session
                .locate_views(VIEW_TYPE, frame.predicted_display_time, &self.local)
                .map_err(|e| format!("OpenXR locate eyes: {e}"))?;
            if validity.contains(
                xr::ViewStateFlags::POSITION_VALID | xr::ViewStateFlags::ORIENTATION_VALID,
            ) {
                // Some runtimes provide valid stereo poses before VIEW-space
                // location becomes valid. Those poses also locate the panel.
                if self.ui_anchor.pose().is_none() && views.len() == 2 {
                    let mut head = views[0].pose;
                    head.position.x = (head.position.x + views[1].pose.position.x) * 0.5;
                    head.position.y = (head.position.y + views[1].pose.position.y) * 0.5;
                    head.position.z = (head.position.z + views[1].pose.position.z) * 0.5;
                    self.ui_anchor.capture(head);
                }
                self.views = views;
                self.eyes.acquire()?;
            }
        }
        Ok(())
    }

    pub fn eye_views(
        &self,
        rig_position: Vec3,
        rig_rotation: Quat,
    ) -> Vec<(wgpu::TextureView, crate::rendering::RenderView)> {
        let (near, far) = controls::clip_planes(self.world_units_per_meter);
        self.views
            .iter()
            .enumerate()
            .filter_map(|(index, eye)| {
                let target = self.eyes.view(index as u32)?;
                let position = Vec3::new(
                    eye.pose.position.x,
                    eye.pose.position.y,
                    eye.pose.position.z,
                );
                let rotation = Quat::from_xyzw(
                    eye.pose.orientation.x,
                    eye.pose.orientation.y,
                    eye.pose.orientation.z,
                    eye.pose.orientation.w,
                );
                Some((
                    target,
                    crate::rendering::RenderView {
                        position: rig_position
                            + rig_rotation * position * self.world_units_per_meter,
                        rotation: (rig_rotation * rotation).normalize(),
                        projection: crate::rendering::CameraProjection::from_fov(
                            eye.fov.angle_left,
                            eye.fov.angle_right,
                            eye.fov.angle_down,
                            eye.fov.angle_up,
                            near,
                            far,
                        ),
                        width: self.eyes.width,
                        height: self.eyes.height,
                    },
                ))
            })
            .collect()
    }

    pub fn ui_view(&self, eye: u32) -> Option<wgpu::TextureView> {
        self.ui.view(eye)
    }

    /// Look through a fixed panel aperture instead of using the headset's full FOV.
    pub fn panel_views(
        &self,
        camera: &crate::ui::camera::CameraController,
        rect: egui::Rect,
        width: u32,
        height: u32,
    ) -> Vec<crate::rendering::RenderView> {
        let Some(pose) = self.ui_anchor.pose() else {
            return Vec::new();
        };
        self.views
            .iter()
            .filter_map(|eye| {
                panel::render_view_with_settings(
                    pose,
                    eye.pose,
                    camera,
                    rect,
                    self.ui.width,
                    self.ui.height,
                    width,
                    height,
                    self.ui_anchor.settings,
                )
            })
            .collect()
    }
    pub fn panel_pointer_ray(
        &self,
        camera: &crate::ui::camera::CameraController,
        pointer: glam::Vec2,
    ) -> Option<(Vec3, Vec3)> {
        if self.views.len() != 2 {
            return None;
        }
        let mut eye = self.views[0].pose;
        let right = self.views[1].pose.position;
        eye.position.x = (eye.position.x + right.x) * 0.5;
        eye.position.y = (eye.position.y + right.y) * 0.5;
        eye.position.z = (eye.position.z + right.z) * 0.5;
        let rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(self.ui.width as f32, self.ui.height as f32),
        );
        let view = panel::render_view_with_settings(
            self.ui_anchor.pose()?,
            eye,
            camera,
            rect,
            self.ui.width,
            self.ui.height,
            self.ui.width,
            self.ui.height,
            self.ui_anchor.settings,
        )?;
        let ndc = Vec3::new(
            pointer.x / self.ui.width as f32 * 2.0 - 1.0,
            1.0 - pointer.y / self.ui.height as f32 * 2.0,
            1.0,
        );
        let ray = view
            .projection
            .matrix(1.0, 0.1, 5000.0)
            .inverse()
            .project_point3(ndc)
            .normalize();
        Some((view.position, view.rotation * ray))
    }
    pub fn head_pose(&self, rig_position: Vec3, rig_rotation: Quat) -> Option<(Vec3, Quat)> {
        let location = self
            .view_space
            .locate(&self.local, self.frame?.predicted_display_time)
            .ok()?;
        if !location.location_flags.contains(
            xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID,
        ) {
            return None;
        }
        let pose = location.pose;
        let position = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
        let rotation = Quat::from_xyzw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        );
        Some((
            rig_position + rig_rotation * position * self.world_units_per_meter,
            (rig_rotation * rotation).normalize(),
        ))
    }
    pub fn fade_world(&self, opacity: f32) {
        if !self.immersive || !self.eyes_drawn {
            return;
        }
        let targets: Vec<_> = (0..2).filter_map(|eye| self.eyes.view(eye)).collect();
        self.fade.draw(&self.device, &self.queue, &targets, opacity);
    }
    pub fn mark_eyes_drawn(&mut self) {
        self.eyes_drawn = true;
    }
    pub fn set_panel_settings(&mut self, settings: PanelSettings) {
        self.ui_anchor.settings = settings.sanitized();
    }
    pub fn mark_ui_drawn(&mut self) {
        self.ui_drawn = true;
        let Some(panel) = self.ui_anchor.pose() else {
            return;
        };
        let clear = !self.eyes_drawn;
        let mut drawn = 0;
        for (index, eye) in self.views.iter().enumerate() {
            let (Some(source), Some(target)) =
                (self.ui.view(index as u32), self.eyes.view(index as u32))
            else {
                continue;
            };
            let projection = crate::rendering::CameraProjection::from_fov(
                eye.fov.angle_left,
                eye.fov.angle_right,
                eye.fov.angle_down,
                eye.fov.angle_up,
                0.005,
                250.0,
            )
            .matrix(1.0, 0.005, 250.0);
            let view = glam::Mat4::from_rotation_translation(
                input::rotation(eye.pose),
                input::position(eye.pose),
            )
            .inverse();
            self.screen_surface.draw(
                &self.device,
                &self.queue,
                &target,
                &source,
                panel,
                self.ui_anchor.settings,
                projection * view,
                clear,
            );
            drawn += 1;
        }
        self.ui_projected = drawn == 2;
        self.eyes_drawn |= self.ui_projected;
    }

    pub fn end_frame(&mut self) -> VrResult<()> {
        let Some(frame) = self.frame.take() else {
            return Ok(());
        };
        // OpenXR requires released Vulkan color images in COLOR_ATTACHMENT_OPTIMAL.
        // Mirroring samples an eye image, so restore its layout before release.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("OpenXR image release"),
            });
        for swapchain in [&self.eyes, &self.ui, &self.wheel, &self.hands] {
            if let Some(index) = swapchain.acquired {
                encoder.transition_resources(
                    std::iter::empty(),
                    std::iter::once(wgpu::TextureTransition {
                        texture: &swapchain.textures[index as usize],
                        selector: None,
                        state: wgpu::TextureUses::COLOR_TARGET,
                    }),
                );
            }
        }
        self.queue.submit([encoder.finish()]);
        let eye_release = self.eyes.release();
        let ui_release = self.ui.release();
        let wheel_release = self.wheel.release();
        let hand_release = self.hands.release();
        let eyes = self.eyes.swapchain.lock().unwrap();
        let ui = self.ui.swapchain.lock().unwrap();
        let wheel = self.wheel.swapchain.lock().unwrap();
        let hands = self.hands.swapchain.lock().unwrap();
        let projection_views: Vec<_> = self
            .views
            .iter()
            .enumerate()
            .map(|(index, view)| {
                xr::CompositionLayerProjectionView::new()
                    .pose(view.pose)
                    .fov(view.fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&eyes)
                            .image_array_index(index as u32)
                            .image_rect(self.eyes.rect()),
                    )
            })
            .collect();
        let projection = xr::CompositionLayerProjection::new()
            .space(&self.local)
            .views(&projection_views);
        let hand_views: Vec<_> = self
            .views
            .iter()
            .enumerate()
            .map(|(index, view)| {
                xr::CompositionLayerProjectionView::new()
                    .pose(view.pose)
                    .fov(view.fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&hands)
                            .image_array_index(index as u32)
                            .image_rect(self.hands.rect()),
                    )
            })
            .collect();
        let hand_projection = xr::CompositionLayerProjection::new()
            .space(&self.local)
            .layer_flags(xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
            .views(&hand_views);
        let quad = |index, visibility| {
            xr::CompositionLayerQuad::new()
                .space(&self.local)
                .layer_flags(xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
                .eye_visibility(visibility)
                .pose(self.ui_anchor.pose().unwrap_or(xr::Posef::IDENTITY))
                .size(xr::Extent2Df {
                    width: panel::WIDTH_METERS,
                    height: panel::WIDTH_METERS / self.ui_anchor.settings.aspect,
                })
                .sub_image(
                    xr::SwapchainSubImage::new()
                        .swapchain(&ui)
                        .image_array_index(index)
                        .image_rect(self.ui.rect()),
                )
        };
        let left_quad = quad(0, xr::EyeVisibility::LEFT);
        let right_quad = quad(1, xr::EyeVisibility::RIGHT);
        let wheel_quad = xr::CompositionLayerQuad::new()
            .space(&self.local)
            .layer_flags(xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
            .eye_visibility(xr::EyeVisibility::BOTH)
            .pose(self.wheel_pose.unwrap_or(xr::Posef::IDENTITY))
            .size(xr::Extent2Df {
                width: controls::WHEEL_METERS,
                height: controls::WHEEL_METERS,
            })
            .sub_image(
                xr::SwapchainSubImage::new()
                    .swapchain(&wheel)
                    .image_array_index(0)
                    .image_rect(self.wheel.rect()),
            );
        let mut layers: Vec<&xr::CompositionLayerBase<'_, xr::Vulkan>> = Vec::new();
        if self.headset_active()
            && eye_release.is_ok()
            && self.eyes_drawn
            && projection_views.len() == 2
        {
            layers.push(&projection);
        }
        if self.headset_active()
            && ui_release.is_ok()
            && self.ui_drawn
            && !self.ui_projected
            && self.ui_anchor.pose().is_some()
        {
            layers.push(&left_quad);
            layers.push(&right_quad);
        }
        // Controllers stay visible over the stationary settings panel.
        if self.headset_active()
            && self.hands_drawn
            && hand_release.is_ok()
            && hand_views.len() == 2
        {
            layers.push(&hand_projection);
        }
        // The wrist console overlays its supporting controller. Its own cursor
        // stays on top of the buttons while the controller remains underneath.
        if self.headset_active() && self.wheel_pose.is_some() && wheel_release.is_ok() {
            layers.push(&wheel_quad);
        }
        self.submitted_frames += 1;
        if self
            .last_frame_report
            .is_none_or(|at| at.elapsed().as_secs() >= 5)
        {
            log::info!(
                "OpenXR frame {}: render={}, active={}, tracked_eyes={}, anchor={:?}, immersive={}, world_drawn={}, ui_drawn={}, layers={}, wheel_drawn={}, hands_drawn={}",
                self.submitted_frames, frame.should_render, self.headset_active(),
                self.views.len(), self.ui_anchor.pose(), self.immersive,
                self.eyes_drawn, self.ui_drawn, layers.len(), self.wheel_pose.is_some(), self.hands_drawn
            );
            self.last_frame_report = Some(std::time::Instant::now());
        }
        let end = self
            .frame_stream
            .end(
                frame.predicted_display_time,
                xr::EnvironmentBlendMode::OPAQUE,
                &layers,
            )
            .map_err(|e| format!("OpenXR end frame: {e}"));
        eye_release
            .and(ui_release)
            .and(wheel_release)
            .and(hand_release)
            .and(end)?;
        // Start waiting for the next headset frame immediately so the runtime
        // wait overlaps app work instead of blocking at the next render entry.
        self.frame_waiter.request(self.session_epoch)
    }

    pub fn recenter_panel(&mut self) {
        self.ui_anchor.reset();
        if let Some(frame) = self.frame {
            if let Ok(head) = self
                .view_space
                .locate(&self.local, frame.predicted_display_time)
            {
                if frame.should_render
                    && head.location_flags.contains(
                        xr::SpaceLocationFlags::POSITION_VALID
                            | xr::SpaceLocationFlags::ORIENTATION_VALID,
                    )
                {
                    self.ui_anchor.capture(head.pose);
                }
            }
        }
    }
    pub fn prepare_ui(&mut self) -> VrResult<()> {
        if self.presenting() && self.views.len() == 2 {
            self.ui.acquire()?;
        }
        Ok(())
    }
    pub fn selection_feedback(&self, selection: bool) {
        if self.focused {
            self.actions.feedback(&self.session, selection);
        }
    }

    pub fn render_controllers(
        &mut self,
        rig: (Vec3, Quat),
        radius: f32,
        controls: &controls::Controls,
        tool: crate::ui::radial_menu::RadialTool,
        drag_distance: Option<f32>,
    ) {
        if !self.presenting() || self.views.len() != 2 {
            return;
        }
        let input = self.tracked_input.borrow().clone();
        if !input
            .grips
            .iter()
            .chain(input.aims.iter())
            .any(Option::is_some)
        {
            return;
        }
        if let Err(error) = self.hands.acquire() {
            log::warn!("OpenXR controller layer: {error}");
            return;
        }
        let ray = input.aims[1]
            .map(|pose| (input::position(pose), input::rotation(pose) * Vec3::NEG_Z))
            .or(input.ray);
        let beam = ray.map(|(tracking_origin, tracking_direction)| {
            let origin = rig.0 + rig.1 * tracking_origin * self.world_units_per_meter;
            let direction = rig.1 * tracking_direction;

            let wrist_menu_hit = controls.wheel_pose.is_some_and(|pose| {
                controls::wheel_hit(pose, tracking_origin, tracking_direction).is_some()
            });
            let panel_hit = controls
                .wheel_pose
                .or(controls.menu_pose.filter(|pose| {
                    controls::wheel_hit(*pose, tracking_origin, tracking_direction)
                        .is_some_and(|p| (p - glam::Vec2::splat(360.0)).length() <= 60.0)
                }))
                .filter(|pose| {
                    controls::wheel_hit(*pose, tracking_origin, tracking_direction).is_some()
                })
                .and_then(|pose| {
                    let normal = input::rotation(pose) * Vec3::Z;
                    let denominator = normal.dot(tracking_direction);
                    if denominator >= -0.001 {
                        return None;
                    }
                    let t = normal.dot(input::position(pose) - tracking_origin) / denominator;
                    (t > 0.0).then_some(t)
                })
                .or_else(|| {
                    if self.immersive && !controls.full_ui {
                        return None;
                    }
                    panel::ray_hit_with_settings(
                        self.ui_anchor.pose()?,
                        self.ui_anchor.settings,
                        tracking_origin,
                        tracking_direction,
                        self.ui.width,
                        self.ui.height,
                    )
                    .map(|hit| hit.1)
                })
                .map(|distance| distance * self.world_units_per_meter);
            let b = origin.dot(direction);
            let discriminant = b * b - origin.length_squared() + radius * radius;
            let sphere_hit = if discriminant >= 0.0 {
                let near = -b - discriminant.sqrt();
                let far = -b + discriminant.sqrt();
                (far > 0.0).then_some(if near > 0.0 { near } else { far })
            } else {
                None
            };
            let tool_distance = if tool == crate::ui::radial_menu::RadialTool::Insert {
                controls::placement_distance(origin, direction, radius, self.world_units_per_meter)
            } else {
                drag_distance
            };
            let distance = panel_hit
                .or(tool_distance)
                .or(sphere_hit)
                .unwrap_or(8.0 * self.world_units_per_meter);
            (
                tracking_origin,
                tracking_origin + tracking_direction * (distance / self.world_units_per_meter),
                !wrist_menu_hit,
            )
        });
        for (eye, view) in self.views.iter().enumerate() {
            let Some(target) = self.hands.view(eye as u32) else {
                continue;
            };
            let projection = crate::rendering::CameraProjection::from_fov(
                view.fov.angle_left,
                view.fov.angle_right,
                view.fov.angle_down,
                view.fov.angle_up,
                0.005,
                250.0,
            )
            .matrix(1.0, 0.005, 250.0);
            let view_matrix = glam::Mat4::from_rotation_translation(
                input::rotation(view.pose),
                input::position(view.pose),
            )
            .inverse();
            self.controller_renderer.draw(
                &self.device,
                &self.queue,
                &target,
                eye,
                projection * view_matrix,
                &input,
                beam,
            );
        }
        self.hands_drawn = true;
    }

    pub fn render_wheel(
        &mut self,
        controls: &controls::Controls,
        tool: crate::ui::radial_menu::RadialTool,
        value_ctx: &egui::Context,
    ) {
        self.wheel_pose = None;
        let pose = controls.wheel_pose.or(controls.menu_pose);
        if pose.is_none() && !(controls.number_pad_active && controls.wheel_pointer_released) {
            return;
        }
        if !self.presenting() {
            return;
        }
        if let Err(error) = self.wheel.acquire() {
            log::warn!("OpenXR wrist wheel: {error}");
            return;
        }
        let Some(target) = self.wheel.view(0) else {
            return;
        };
        let size = controls::WHEEL_PIXELS;
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size as f32, size as f32),
            )),
            ..Default::default()
        };
        if controls.number_pad_active {
            if let Some(pointer) = controls.pointer {
                let pos = egui::pos2(pointer.x, pointer.y);
                input.events.push(egui::Event::PointerMoved(pos));
                if controls.wheel_pointer_pressed {
                    input.events.push(egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: egui::Modifiers::NONE,
                    });
                }
            }
            if controls.wheel_pointer_released {
                let pos = controls
                    .pointer
                    .map(|pointer| egui::pos2(pointer.x, pointer.y))
                    .unwrap_or(egui::pos2(-1_000_000.0, -1_000_000.0));
                input.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                });
            }
        }
        if let Some(viewport) = input.viewports.get_mut(&input.viewport_id) {
            viewport.native_pixels_per_point = Some(self.wheel_pixel_scale);
        }
        self.wheel_ctx.begin_pass(input);
        if controls.number_pad_active {
            let pointer = controls
                .pointer
                .map(|pointer| egui::pos2(pointer.x, pointer.y));
            egui::ControllerNumberPad::show_on_controller(&self.wheel_ctx, value_ctx, pointer);
        } else {
            controls.draw(&self.wheel_ctx, tool);
        }
        let output = self.wheel_ctx.end_pass();
        let jobs = self
            .wheel_ctx
            .tessellate(output.shapes, output.pixels_per_point);
        for (id, delta) in &output.textures_delta.set {
            self.wheel_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.wheel.width, self.wheel.height],
            pixels_per_point: self.wheel_pixel_scale,
        };
        let buffers = self.wheel_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &jobs,
            &screen,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Left hand wheel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.wheel_renderer
                .render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        self.queue
            .submit(buffers.into_iter().chain([encoder.finish()]));
        for id in output.textures_delta.free {
            self.wheel_renderer.free_texture(&id);
        }
        self.wheel_pose = pose;
    }
}

impl Drop for VrState {
    fn drop(&mut self) {
        let _ = self.end_frame();
        if self.running {
            let _ = self.session.request_exit();
        }
        // The runtime owns the imported images, so complete queued work before releasing them.
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }
}

pub fn runtime_info() -> VrResult<String> {
    runtime::connect(runtime_info_current)
}

fn runtime_info_current() -> VrResult<String> {
    let (instance, refresh) = create_instance()?;
    let properties = instance
        .properties()
        .map_err(|e| format!("OpenXR runtime properties: {e}"))?;
    let headset = match instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY) {
        Ok(system) => instance
            .system_properties(system)
            .map(|p| p.system_name)
            .map_err(|e| e.to_string())?,
        Err(error) => {
            return Err(format!(
                "Headset unavailable ({error}); connect the headset and start PCVR"
            ))
        }
    };
    Ok(format!("Runtime: {} {}\nHeadset: {headset}\nVulkan enable2: true\nDisplay refresh control: {refresh}", properties.runtime_name, properties.runtime_version))
}

/// Exercise runtime-selected GPU creation and imported swapchains without opening a window.
pub fn check_graphics() -> VrResult<String> {
    let bootstrap = VrBootstrap::new()?;
    let graphics = bootstrap.create_graphics()?;
    // Verify a real GPU submission and its synchronization before creating a
    // session. Swapchain creation alone missed the original wait_semaphores crash.
    let encoder = graphics
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Native VR synchronization check"),
        });
    let submission = graphics.queue.submit([encoder.finish()]);
    graphics
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .map_err(|e| format!("OpenXR GPU synchronization check: {e}"))?;
    let state = VrState::new(
        bootstrap,
        &graphics,
        &[
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        ],
        1280,
        720,
    )?;
    let info = format!(
        "OpenXR Vulkan GPU submission, session and swapchains verified on {}. Eye resolution: {:?}",
        graphics.adapter.get_info().name,
        state.dimensions()
    );
    drop(state);
    Ok(info)
}

/// The Windows release executable has no console, so startup failures need a visible explanation.
pub fn show_startup_error(error: &str) {
    eprintln!("Native VR: {error}");
    #[cfg(target_os = "windows")]
    {
        let title: Vec<u16> = "Biospheres VR".encode_utf16().chain(Some(0)).collect();
        let body: Vec<u16> = format!("Native VR could not start.\n\n{error}\n\nConnect your headset, start PCVR, and check the active OpenXR runtime.").encode_utf16().chain(Some(0)).collect();
        // SAFETY: both null-terminated UTF-16 strings remain alive during this call.
        unsafe {
            winapi::um::winuser::MessageBoxW(
                std::ptr::null_mut(),
                body.as_ptr(),
                title.as_ptr(),
                winapi::um::winuser::MB_OK | winapi::um::winuser::MB_ICONERROR,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_supersampling_doubles_dimensions_when_device_limits_allow() {
        assert_eq!(supersampled_dimensions(1920, 1080, 4096), (3840, 2160, 2.0));
    }

    #[test]
    fn ui_supersampling_respects_texture_dimension_limits() {
        let (width, height, scale) = supersampled_dimensions(3000, 2000, 4096);
        assert!(width <= 4096);
        assert!(height <= 4096);
        assert!(scale > 1.0 && scale < UI_SUPERSAMPLE_FACTOR);
        assert!((width as f32 / height as f32 - 1.5).abs() < 0.001);
    }

    #[test]
    fn selects_vulkan_version_supported_by_runtime_and_loader() {
        assert_eq!(
            select_vulkan_api_version(
                xr::Version::new(1, 0, 0),
                xr::Version::new(1, 2, 0),
                vk::API_VERSION_1_3,
            )
            .unwrap(),
            vk::API_VERSION_1_2,
        );
        assert_eq!(
            select_vulkan_api_version(
                xr::Version::new(1, 0, 0),
                xr::Version::new(1, 3, 0),
                vk::API_VERSION_1_2,
            )
            .unwrap(),
            vk::API_VERSION_1_2,
        );
        assert!(select_vulkan_api_version(
            xr::Version::new(1, 0, 0),
            xr::Version::new(1, 1, 0),
            vk::API_VERSION_1_3,
        )
        .is_err());
    }

    /// Exercise the core function that failed on the headset GPU without needing
    /// an OpenXR session. Timeline semaphore waits are core in Vulkan 1.2.
    #[test]
    fn native_vulkan_api_exposes_working_timeline_semaphore_waits() {
        struct InstanceOwner(ash::Instance);
        impl Drop for InstanceOwner {
            fn drop(&mut self) {
                unsafe { self.0.destroy_instance(None) };
            }
        }
        struct DeviceOwner(ash::Device);
        impl Drop for DeviceOwner {
            fn drop(&mut self) {
                unsafe { self.0.destroy_device(None) };
            }
        }
        // SAFETY: owners destroy device before instance, and the loader outlives
        // both. The semaphore has no outstanding queue operations when destroyed.
        unsafe {
            let entry = ash::Entry::load().expect("Vulkan loader required for native VR test");
            let app = vk::ApplicationInfo::default().api_version(MINIMUM_VULKAN_API_VERSION);
            let instance = InstanceOwner(
                entry
                    .create_instance(
                        &vk::InstanceCreateInfo::default().application_info(&app),
                        None,
                    )
                    .expect("native Vulkan instance"),
            );
            let (physical, family) = instance
                .0
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find_map(|physical| {
                    if instance
                        .0
                        .get_physical_device_properties(physical)
                        .api_version
                        < vk::API_VERSION_1_2
                    {
                        return None;
                    }
                    instance
                        .0
                        .get_physical_device_queue_family_properties(physical)
                        .iter()
                        .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                        .map(|family| (physical, family as u32))
                })
                .expect("Vulkan GPU with core timeline semaphores");
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let mut timeline =
                vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
            let device = DeviceOwner(
                instance
                    .0
                    .create_device(
                        physical,
                        &vk::DeviceCreateInfo::default()
                            .queue_create_infos(&queues)
                            .push_next(&mut timeline),
                        None,
                    )
                    .expect("native Vulkan device"),
            );
            assert!(
                instance
                    .0
                    .get_device_proc_addr(device.0.handle(), c"vkWaitSemaphores".as_ptr())
                    .is_some(),
                "The declared Vulkan version must expose the core wait entry point"
            );
            let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(7);
            let semaphore = device
                .0
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info),
                    None,
                )
                .unwrap();
            let semaphores = [semaphore];
            let values = [7];
            let result = device.0.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values),
                5_000_000_000,
            );
            device.0.destroy_semaphore(semaphore, None);
            result.expect("timeline synchronization must work on the native Vulkan device");
        }
    }
}
