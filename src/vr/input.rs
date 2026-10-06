use glam::{Quat, Vec2, Vec3};
use openxr as xr;

#[derive(Default)]
pub struct VrInput {
    pub pointer: Option<Vec2>,
    pub ray: Option<(Vec3, Vec3)>,
    pub select: bool,
    pub menu: bool,
    pub movement: Vec2,
    pub turn: f32,
    pub head_rotation: Option<Quat>,
}

pub(super) struct Actions {
    spaces: [xr::Space; 2],
    aim: xr::Action<xr::Posef>,
    trigger: xr::Action<f32>,
    click: xr::Action<bool>,
    menu: xr::Action<bool>,
    stick: xr::Action<xr::Vector2f>,
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
            let trigger = set.create_action::<f32>("trigger", "Select or use tool", &hands)?;
            let click = set.create_action::<bool>("click", "Select", &hands)?;
            let menu = set.create_action::<bool>("tools", "Tool menu", &hands)?;
            let stick = set.create_action::<xr::Vector2f>("stick", "Move and snap turn", &hands)?;
            for (profile, select, stick_path, menu_left, menu_right) in [
                (
                    "oculus/touch_controller",
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
                let mut bindings = Vec::new();
                for (index, hand) in ["left", "right"].into_iter().enumerate() {
                    let path = |component: &str| {
                        instance.string_to_path(&format!("/user/hand/{hand}/input/{component}"))
                    };
                    bindings.push(xr::Binding::new(&aim, path("aim/pose")?));
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
                spaces,
                aim,
                trigger,
                click,
                menu,
                stick,
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
        width: u32,
        height: u32,
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
            }
        }
        for index in [1, 0] {
            let path = self.hands[index];
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
            let mut pointer = None;
            if let Ok(location) = self.spaces[index].locate(head, time) {
                if location.location_flags.contains(
                    xr::SpaceLocationFlags::POSITION_VALID
                        | xr::SpaceLocationFlags::ORIENTATION_VALID,
                ) {
                    pointer = panel_intersection(
                        position(location.pose),
                        rotation(location.pose) * Vec3::NEG_Z,
                        width,
                        height,
                    );
                }
            }
            let select = self
                .trigger
                .state(session, path)
                .is_ok_and(|s| s.is_active && s.current_state > 0.55)
                || self
                    .click
                    .state(session, path)
                    .is_ok_and(|s| s.is_active && s.current_state);
            if result.ray.is_none() || pointer.is_some() {
                result.ray = ray;
                result.pointer = pointer;
                result.select = select;
            }
            if pointer.is_some() {
                break;
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
            }
        }
        Ok(result)
    }
}

fn position(pose: xr::Posef) -> Vec3 {
    Vec3::new(pose.position.x, pose.position.y, pose.position.z)
}
fn rotation(pose: xr::Posef) -> Quat {
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
    if direction.z >= -1e-5 || width == 0 || height == 0 {
        return None;
    }
    let distance = (-1.6 - origin.z) / direction.z;
    if distance < 0.0 {
        return None;
    }
    let point = origin + direction * distance;
    let panel_height = 1.8 * height as f32 / width as f32;
    let uv = Vec2::new(point.x / 1.8 + 0.5, 0.5 - point.y / panel_height);
    if !uv.is_finite() || uv.min_element() < 0.0 || uv.max_element() > 1.0 {
        return None;
    }
    Some(uv * Vec2::new(width as f32, height as f32))
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
