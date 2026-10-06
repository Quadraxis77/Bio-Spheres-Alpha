//! Native OpenXR presentation using the same Vulkan device as the wgpu renderer.

use ash::vk::{self, Handle};
use glam::{Quat, Vec3};
use openxr as xr;
use std::sync::{Arc, Mutex};
pub mod input;
mod mirror;

pub type VrResult<T> = Result<T, String>;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
type Vulkan = wgpu::hal::api::Vulkan;

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
    binding: xr::vulkan::SessionCreateInfo,
}

impl VrBootstrap {
    pub fn new() -> VrResult<Self> {
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
        let version = xr::Version::new(1, 1, 0);
        if version < requirements.min_api_version_supported {
            return Err(format!("The OpenXR runtime requires Vulkan {}, above this integration's Vulkan 1.1 target.", requirements.min_api_version_supported));
        }
        // SAFETY: all Vulkan handles are created through OpenXR, then handed to
        // wgpu-hal exactly once. ash clones are borrowed function tables, not owners.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader: {e}"))?;
            let api_version = vk::API_VERSION_1_1;
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
                binding: xr::vulkan::SessionCreateInfo {
                    instance: instance_handle.as_raw() as _,
                    physical_device: physical.as_raw() as _,
                    device: device_handle.as_raw() as _,
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
    eyes: RuntimeSwapchain,
    ui: RuntimeSwapchain,
    frame_stream: xr::FrameStream<xr::Vulkan>,
    frame_waiter: xr::FrameWaiter,
    local: xr::Space,
    view_space: xr::Space,
    session: xr::Session<xr::Vulkan>,
    instance: xr::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    running: bool,
    focused: bool,
    exiting: bool,
    frame: Option<xr::FrameState>,
    views: Vec<xr::View>,
    pub world_units_per_meter: f32,
    eyes_drawn: bool,
    ui_drawn: bool,
    format: wgpu::TextureFormat,
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
            bootstrap
                .instance
                .create_session::<xr::Vulkan>(bootstrap.system, &graphics.binding)
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
        let ui = RuntimeSwapchain::new(
            &session,
            &graphics.device,
            format,
            ui_width.max(1),
            ui_height.max(1),
            1,
        )?;
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
                }
                log::info!("OpenXR refresh rates: {rates:?}");
            }
        }
        log::info!("OpenXR eye resolution: {width} x {height}");
        Ok(Self {
            actions,
            mirror: mirror::Mirror::new(&graphics.device, format),
            eyes,
            ui,
            frame_stream,
            frame_waiter,
            local,
            view_space,
            session,
            instance: bootstrap.instance,
            device: graphics.device.clone(),
            queue: graphics.queue.clone(),
            running: false,
            focused: false,
            exiting: false,
            frame: None,
            views: vec![],
            world_units_per_meter: 20.0,
            eyes_drawn: false,
            ui_drawn: false,
            format,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.eyes.width, self.eyes.height)
    }
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
    pub fn running(&self) -> bool {
        self.running
    }
    pub fn frame_active(&self) -> bool {
        self.frame.is_some()
    }
    pub fn should_exit(&self) -> bool {
        self.exiting
    }
    pub fn input(&self) -> VrResult<input::VrInput> {
        let Some(frame) = self.frame else {
            return Ok(Default::default());
        };
        self.actions.sample(
            &self.session,
            &self.local,
            &self.view_space,
            frame.predicted_display_time,
            self.focused && frame.should_render,
            self.ui.width,
            self.ui.height,
        )
    }
    pub fn resize_ui(&mut self, width: u32, height: u32) -> VrResult<()> {
        if self.frame_active() {
            return Err("Cannot resize an OpenXR swapchain during a frame".into());
        }
        if (self.ui.width, self.ui.height) == (width, height) {
            return Ok(());
        }
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| e.to_string())?;
        self.ui = RuntimeSwapchain::new(
            &self.session,
            &self.device,
            self.format,
            width.max(1),
            height.max(1),
            1,
        )?;
        Ok(())
    }
    pub fn render_mirror(&self, queue: &wgpu::Queue, target: &wgpu::TextureView) {
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
                    match event.state() {
                        xr::SessionState::READY => {
                            self.session
                                .begin(VIEW_TYPE)
                                .map_err(|e| format!("OpenXR begin session: {e}"))?;
                            self.running = true;
                        }
                        xr::SessionState::STOPPING => {
                            self.session
                                .end()
                                .map_err(|e| format!("OpenXR end session: {e}"))?;
                            self.running = false;
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            self.exiting = true
                        }
                        _ => {}
                    }
                }
                xr::Event::InstanceLossPending(_) => self.exiting = true,
                _ => {}
            }
        }
        if !self.running || self.exiting {
            return Ok(());
        }
        let frame = self
            .frame_waiter
            .wait()
            .map_err(|e| format!("OpenXR wait frame: {e}"))?;
        self.frame_stream
            .begin()
            .map_err(|e| format!("OpenXR begin frame: {e}"))?;
        self.frame = Some(frame);
        self.eyes_drawn = false;
        self.ui_drawn = false;
        self.views.clear();
        if frame.should_render {
            let (validity, views) = self
                .session
                .locate_views(VIEW_TYPE, frame.predicted_display_time, &self.local)
                .map_err(|e| format!("OpenXR locate eyes: {e}"))?;
            if validity.contains(
                xr::ViewStateFlags::POSITION_VALID | xr::ViewStateFlags::ORIENTATION_VALID,
            ) {
                self.views = views;
                self.eyes.acquire()?;
                self.ui.acquire()?;
            }
        }
        Ok(())
    }

    pub fn eye_views(
        &self,
        rig_position: Vec3,
        rig_rotation: Quat,
    ) -> Vec<(wgpu::TextureView, crate::rendering::RenderView)> {
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
                            0.1,
                            5000.0,
                        ),
                        width: self.eyes.width,
                        height: self.eyes.height,
                    },
                ))
            })
            .collect()
    }

    pub fn ui_view(&self) -> Option<wgpu::TextureView> {
        self.ui.view(0)
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
    pub fn mark_eyes_drawn(&mut self) {
        self.eyes_drawn = true;
    }
    pub fn mark_ui_drawn(&mut self) {
        self.ui_drawn = true;
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
        for swapchain in [&self.eyes, &self.ui] {
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
        let eyes = self.eyes.swapchain.lock().unwrap();
        let ui = self.ui.swapchain.lock().unwrap();
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
        let quad = xr::CompositionLayerQuad::new()
            .space(&self.view_space)
            .layer_flags(xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
            .eye_visibility(xr::EyeVisibility::BOTH)
            .pose(xr::Posef {
                orientation: xr::Quaternionf {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                position: xr::Vector3f {
                    x: 0.0,
                    y: 0.0,
                    z: -1.6,
                },
            })
            .size(xr::Extent2Df {
                width: 1.8,
                height: 1.8 * self.ui.height as f32 / self.ui.width as f32,
            })
            .sub_image(
                xr::SwapchainSubImage::new()
                    .swapchain(&ui)
                    .image_array_index(0)
                    .image_rect(self.ui.rect()),
            );
        let mut layers: Vec<&xr::CompositionLayerBase<'_, xr::Vulkan>> = Vec::new();
        if eye_release.is_ok() && self.eyes_drawn && projection_views.len() == 2 {
            layers.push(&projection);
        }
        if ui_release.is_ok() && self.ui_drawn {
            layers.push(&quad);
        }
        let end = self
            .frame_stream
            .end(
                frame.predicted_display_time,
                xr::EnvironmentBlendMode::OPAQUE,
                &layers,
            )
            .map_err(|e| format!("OpenXR end frame: {e}"));
        eye_release.and(ui_release).and(end)
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
    let (instance, refresh) = create_instance()?;
    let properties = instance
        .properties()
        .map_err(|e| format!("OpenXR runtime properties: {e}"))?;
    let headset = match instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY) {
        Ok(system) => instance
            .system_properties(system)
            .map(|p| p.system_name)
            .map_err(|e| e.to_string())?,
        Err(error) => format!("Unavailable ({error}); connect the headset and start PCVR"),
    };
    Ok(format!("Runtime: {} {}\nHeadset: {headset}\nVulkan enable2: true\nDisplay refresh control: {refresh}", properties.runtime_name, properties.runtime_version))
}

/// Exercise runtime-selected GPU creation and imported swapchains without opening a window.
pub fn check_graphics() -> VrResult<String> {
    let bootstrap = VrBootstrap::new()?;
    let graphics = bootstrap.create_graphics()?;
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
        "OpenXR Vulkan session and swapchains created on {}. Eye resolution: {:?}",
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
