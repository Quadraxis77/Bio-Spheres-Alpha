use glam::{Quat, Vec2, Vec3};
use openxr as xr;

#[derive(Default, Clone)]
pub struct VrInput {
    pub pointer: Option<Vec2>,
    pub ray: Option<(Vec3, Vec3)>,
    pub select: bool,
    pub menu: bool,
    pub movement: Vec2,
    pub turn: f32,
    pub head_rotation: Option<Quat>,
    pub head_position: Option<Vec3>,
    pub grips: [Option<xr::Posef>; 2],
    pub aims: [Option<xr::Posef>; 2],
    pub squeeze: [bool; 2],
    pub triggers: [bool; 2],
    pub menus: [bool; 2],
    pub lift: f32,
    pub ui_pointer: bool,
    pub wheel_button: bool,
    pub stick_clicks: [bool; 2],
}

pub(super) struct Actions {
    spaces: [xr::Space; 2],
    grip_spaces: [xr::Space; 2],
    grip: xr::Action<xr::Posef>,
    squeeze: xr::Action<f32>,
    squeeze_click: xr::Action<bool>,
    haptic: xr::Action<xr::Haptic>,
    aim: xr::Action<xr::Posef>,
    trigger: xr::Action<f32>,
    click: xr::Action<bool>,
    menu: xr::Action<bool>,
    stick: xr::Action<xr::Vector2f>,
    wheel: xr::Action<bool>,
    stick_click: xr::Action<bool>,
    hands: [xr::Path; 2],
    set: xr::ActionSet,
}

impl Actions {
    pub fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
    ) -> super::VrResult<Self> {
        let result = || -> xr::Result<Self> {
            let hands = [
                instance.string_to_path("/user/hand/left")?,
                instance.string_to_path("/user/hand/right")?,
            ];
            let set = instance.create_action_set("biospheres", "Biospheres controls", 0)?;
            let aim = set.create_action::<xr::Posef>("aim", "Point", &hands)?;
            let grip = set.create_action::<xr::Posef>("grip_pose", "Hand position", &hands)?;
            let squeeze = set.create_action::<f32>("squeeze", "Grab world", &hands)?;
            let squeeze_click =
                set.create_action::<bool>("squeeze_click", "Grab world button", &hands)?;
            let haptic =
                set.create_action::<xr::Haptic>("feedback", "Selection feedback", &hands)?;
            let trigger = set.create_action::<f32>("trigger", "Select or use tool", &hands)?;
            let click = set.create_action::<bool>("click", "Select", &hands)?;
            let menu = set.create_action::<bool>("tools", "Tool menu", &hands)?;
            let stick = set.create_action::<xr::Vector2f>("stick", "Pan zoom and orbit", &hands)?;
            let wheel = set.create_action::<bool>("wheel", "Left hand tool wheel", &hands)?;
            let stick_click =
                set.create_action::<bool>("stick_click", "Controller stick click", &hands)?;
            for (profile, select, stick_path, menu_left, menu_right) in [
                (
                    "oculus/touch_controller",
                    "trigger/value",
                    Some("thumbstick"),
                    "y/click",
                    "b/click",
                ),
                (
                    "meta/touch_controller_plus",
                    "trigger/value",
                    Some("thumbstick"),
                    "y/click",
                    "b/click",
                ),
                (
                    "facebook/touch_controller_pro",
                    "trigger/value",
                    Some("thumbstick"),
                    "y/click",
                    "b/click",
                ),
                (
                    "valve/index_controller",
                    "trigger/value",
                    Some("thumbstick"),
                    "b/click",
                    "b/click",
                ),
                (
                    "microsoft/motion_controller",
                    "trigger/value",
                    Some("thumbstick"),
                    "menu/click",
                    "menu/click",
                ),
                (
                    "htc/vive_controller",
                    "trigger/value",
                    Some("trackpad"),
                    "menu/click",
                    "menu/click",
                ),
                (
                    "khr/simple_controller",
                    "select/click",
                    None,
                    "menu/click",
                    "menu/click",
                ),
            ] {
                if (profile == "meta/touch_controller_plus"
                    && instance.exts().meta_touch_controller_plus.is_none())
                    || (profile == "facebook/touch_controller_pro"
                        && instance.exts().fb_touch_controller_pro.is_none())
                {
                    continue;
                }
                let mut bindings = Vec::new();
                for (index, hand) in ["left", "right"].into_iter().enumerate() {
                    let path = |component: &str| {
                        instance.string_to_path(&format!("/user/hand/{hand}/input/{component}"))
                    };
                    bindings.push(xr::Binding::new(&aim, path("aim/pose")?));
                    bindings.push(xr::Binding::new(&grip, path("grip/pose")?));
                    bindings.push(xr::Binding::new(
                        &haptic,
                        instance.string_to_path(&format!("/user/hand/{hand}/output/haptic"))?,
                    ));
                    match profile {
                        "oculus/touch_controller"
                        | "meta/touch_controller_plus"
                        | "facebook/touch_controller_pro"
                        | "valve/index_controller" => {
                            bindings.push(xr::Binding::new(&squeeze, path("squeeze/value")?))
                        }
                        "microsoft/motion_controller" | "htc/vive_controller" => {
                            bindings.push(xr::Binding::new(&squeeze_click, path("squeeze/click")?))
                        }
                        _ => {}
                    }
                    if select.ends_with("value") {
                        bindings.push(xr::Binding::new(&trigger, path(select)?));
                    } else {
                        bindings.push(xr::Binding::new(&click, path(select)?));
                    }
                    bindings.push(xr::Binding::new(
                        &menu,
                        path(if index == 0 { menu_left } else { menu_right })?,
                    ));
                    if let Some(component) = stick_path {
                        bindings.push(xr::Binding::new(&stick, path(component)?));
                        bindings.push(xr::Binding::new(
                            &stick_click,
                            path(&format!("{component}/click"))?,
                        ));
                    }
                    if index == 0 {
                        let button = match profile {
                            "oculus/touch_controller"
                            | "meta/touch_controller_plus"
                            | "facebook/touch_controller_pro" => Some("x/click"),
                            "valve/index_controller" => Some("a/click"),
                            _ => None,
                        };
                        if let Some(button) = button {
                            bindings.push(xr::Binding::new(&wheel, path(button)?));
                        }
                    }
                }
                if let Err(error) = instance.suggest_interaction_profile_bindings(
                    instance.string_to_path(&format!("/interaction_profiles/{profile}"))?,
                    &bindings,
                ) {
                    log::warn!("OpenXR controller bindings for {profile}: {error}");
                }
            }
            session.attach_action_sets(&[&set])?;
            let spaces = [
                aim.create_space(session, hands[0], xr::Posef::IDENTITY)?,
                aim.create_space(session, hands[1], xr::Posef::IDENTITY)?,
            ];
            Ok(Self {
                grip_spaces: [
                    grip.create_space(session, hands[0], xr::Posef::IDENTITY)?,
                    grip.create_space(session, hands[1], xr::Posef::IDENTITY)?,
                ],
                grip,
                squeeze,
                squeeze_click,
                haptic,
                spaces,
                aim,
                trigger,
                click,
                menu,
                stick,
                wheel,
                stick_click,
                hands,
                set,
            })
        };
        result().map_err(|error| format!("OpenXR controllers: {error}"))
    }

    pub fn sample(
        &self,
        session: &xr::Session<xr::Vulkan>,
        local: &xr::Space,
        head: &xr::Space,
        time: xr::Time,
        focused: bool,
        panel_pose: Option<xr::Posef>,
        width: u32,
        height: u32,
        settings: super::PanelSettings,
    ) -> super::VrResult<VrInput> {
        if !focused {
            return Ok(VrInput::default());
        }
        session
            .sync_actions(&[xr::ActiveActionSet::new(&self.set)])
            .map_err(|e| format!("OpenXR controller sync: {e}"))?;
        let mut result = VrInput::default();
        if let Ok(location) = head.locate(local, time) {
            if location
                .location_flags
                .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
            {
                result.head_rotation = Some(rotation(location.pose));
                if location
                    .location_flags
                    .contains(xr::SpaceLocationFlags::POSITION_VALID)
                {
                    result.head_position = Some(position(location.pose));
                }
            }
        }
        for index in [1, 0] {
            let path = self.hands[index];
            if self.grip.is_active(session, path).unwrap_or(false) {
                if let Ok(location) = self.grip_spaces[index].locate(local, time) {
                    if location.location_flags.contains(
                        xr::SpaceLocationFlags::POSITION_VALID
                            | xr::SpaceLocationFlags::ORIENTATION_VALID,
                    ) {
                        result.grips[index] = Some(location.pose);
                    }
                }
            }
            result.squeeze[index] = self
                .squeeze
                .state(session, path)
                .is_ok_and(|s| s.is_active && s.current_state > 0.65)
                || self
                    .squeeze_click
                    .state(session, path)
                    .is_ok_and(|s| s.is_active && s.current_state);
            result.triggers[index] = self
                .trigger
                .state(session, path)
                .is_ok_and(|s| s.is_active && s.current_state > 0.55)
                || self
                    .click
                    .state(session, path)
                    .is_ok_and(|s| s.is_active && s.current_state);
            result.menus[index] = self
                .menu
                .state(session, path)
                .is_ok_and(|s| s.is_active && s.current_state);
            result.stick_clicks[index] = self
                .stick_click
                .state(session, path)
                .is_ok_and(|s| s.is_active && s.current_state);
            if index == 0 {
                result.wheel_button = self
                    .wheel
                    .state(session, path)
                    .is_ok_and(|s| s.is_active && s.current_state);
            }
            if !self.aim.is_active(session, path).unwrap_or(false) {
                continue;
            }
            let location = self.spaces[index]
                .locate(local, time)
                .map_err(|e| format!("OpenXR controller pose: {e}"))?;
            if !location.location_flags.contains(
                xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID,
            ) {
                continue;
            }
            let ray = Some((
                position(location.pose),
                rotation(location.pose) * Vec3::NEG_Z,
            ));
            result.aims[index] = Some(location.pose);
            let pointer = panel_pose.and_then(|pose| {
                super::panel::ray_hit_with_settings(
                    pose,
                    settings,
                    position(location.pose),
                    rotation(location.pose) * Vec3::NEG_Z,
                    width,
                    height,
                )
                .map(|hit| hit.0)
            });
            let select = self
                .trigger
                .state(session, path)
                .is_ok_and(|s| s.is_active && s.current_state > 0.55)
                || self
                    .click
                    .state(session, path)
                    .is_ok_and(|s| s.is_active && s.current_state);
            if result.ray.is_none()
                || (result.pointer.is_none() && pointer.is_some())
                || (select && pointer.is_some() && !result.select)
            {
                result.ray = ray;
                result.pointer = pointer;
                result.select = select;
            }
        }
        for path in self.hands {
            result.menu |= self
                .menu
                .state(session, path)
                .is_ok_and(|s| s.is_active && s.current_state);
        }
        if let Ok(stick) = self.stick.state(session, self.hands[0]) {
            if stick.is_active {
                result.movement = Vec2::new(stick.current_state.x, stick.current_state.y);
            }
        }
        if let Ok(stick) = self.stick.state(session, self.hands[1]) {
            if stick.is_active {
                result.turn = stick.current_state.x;
                result.lift = stick.current_state.y;
            }
        }
        Ok(result)
    }

    pub fn feedback(&self, session: &xr::Session<xr::Vulkan>, selection: bool) {
        let vibration = xr::HapticVibration::new()
            .amplitude(if selection { 0.4 } else { 0.12 })
            .duration(xr::Duration::from_nanos(if selection {
                30_000_000
            } else {
                12_000_000
            }))
            .frequency(xr::FREQUENCY_UNSPECIFIED);
        let _ = self
            .haptic
            .apply_feedback(session, self.hands[1], &vibration);
    }
}

pub(super) fn position(pose: xr::Posef) -> Vec3 {
    Vec3::new(pose.position.x, pose.position.y, pose.position.z)
}
pub(super) fn rotation(pose: xr::Posef) -> Quat {
    Quat::from_xyzw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    .normalize()
}

/// The compositor panel and its pointer use the same physical dimensions.
pub fn panel_intersection(origin: Vec3, direction: Vec3, width: u32, height: u32) -> Option<Vec2> {
    let mut anchor = super::panel::PanelAnchor::default();
    anchor.capture(xr::Posef::IDENTITY);
    super::panel::ray_hit(anchor.pose().unwrap(), origin, direction, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pointer_matches_native_panel_coordinates() {
        assert_eq!(
            panel_intersection(Vec3::ZERO, Vec3::NEG_Z, 1920, 1080),
            Some(Vec2::new(960.0, 540.0))
        );
        assert!(panel_intersection(Vec3::ZERO, Vec3::Z, 1920, 1080).is_none());
        assert!(panel_intersection(Vec3::new(2.0, 0.0, 0.0), Vec3::NEG_Z, 1920, 1080).is_none());
        let pointer =
            panel_intersection(Vec3::new(0.45, 0.253125, 0.0), Vec3::NEG_Z, 1920, 1080).unwrap();
        assert!((pointer - Vec2::new(1440.0, 270.0)).length() < 0.01);
        assert!(panel_intersection(Vec3::new(0.0, 0.0, -2.0), Vec3::NEG_Z, 1920, 1080).is_none());
    }
}
