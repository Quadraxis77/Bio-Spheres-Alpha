//! Lightweight runtime audio for music and short game SFX.
//!
//! Built on kira instead of rodio: kira is a real-time-safe game audio engine
//! (lock-free command queue to the audio thread, fixed-capacity resource
//! pools) built specifically for the "many dynamic short-lived sounds"
//! pattern this file needs - rodio's simpler channel-based mixer could not
//! sustain the churn of hundreds of concurrent, rapidly-replaced voices
//! without the whole output glitching.

use glam::{Quat, Vec3};
use kira::{
    backend::DefaultBackend,
    effect::{
        delay::DelayBuilder,
        filter::{FilterBuilder, FilterHandle},
        volume_control::{VolumeControlBuilder, VolumeControlHandle},
    },
    sound::{
        static_sound::{StaticSoundData, StaticSoundHandle},
        PlaybackState,
    },
    track::{SendTrackBuilder, SendTrackHandle, TrackBuilder, TrackHandle},
    AudioManager, AudioManagerSettings, Decibels, Easing, Mix, Panning, PlaybackRate, Tween,
};
use std::{
    collections::VecDeque,
    io::Cursor,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const MAX_SFX_VOICES: usize = 256;
// Independent of MAX_SFX_VOICES: that bounds how many voices can be
// simultaneously *alive*, this bounds how fast new ones can be *created*.
// Confirmed via logs (under the old rodio backend) that the actual failure
// mode wasn't having 256 voices alive at once - it was connect/disconnect
// churn when a sustained division burst was creating/evicting on the order of
// 1000+ voices/sec. Kept as a second line of defense even on kira's more
// robust engine: a burst can still fill the full 256-voice pool instantly
// (the token bucket's capacity equals MAX_SFX_VOICES), it just can't keep
// *replacing* them faster than this sustained rate.
const MAX_VOICE_CREATES_PER_SEC: f32 = 200.0;
// Kept well above the old 20ms: a new cluster flush means a fresh round of
// voice creates, and firing that every 20ms under a sustained growth burst is
// what took audio down entirely before. Latency cost is real but small next
// to the mixer falling over.
const CELL_DIVIDE_COALESCE_WINDOW: Duration = Duration::from_millis(80);
const MAX_PENDING_CELL_DIVISION_EVENTS: usize = 250_000;
const CELL_DIVIDE_FULL_VOLUME_RADIUS: f32 = 14.0;
const CELL_DIVIDE_AUDIBLE_RADIUS: f32 = 330.0;
const CELL_DIVIDE_FAR_GAIN: f32 = 0.08;
// Beyond CELL_DIVIDE_AUDIBLE_RADIUS the old curve held flat at CELL_DIVIDE_FAR_GAIN
// forever - a division at the far side of the world was exactly as loud as one just
// past the audible radius. This second radius is where that floor finishes tapering
// to true silence instead of staying pinned at the floor indefinitely.
const CELL_DIVIDE_SILENCE_RADIUS: f32 = 600.0;
const UNDERWATER_CELL_DIVIDE_AUDIBLE_RADIUS: f32 = 270.0;
const UNDERWATER_CELL_DIVIDE_FAR_GAIN: f32 = 0.05;
const UNDERWATER_CELL_DIVIDE_SILENCE_RADIUS: f32 = 480.0;
// Real water absorbs high frequencies far more than low ones - this is the actual
// physical reason whale calls (mostly under a few hundred Hz) carry for miles while
// a sharp transient like a snap or click dies out within meters. Modeled as a
// low-pass filter whose cutoff drops with distance, not a flat volume penalty.
const UNDERWATER_LOWPASS_NEAR_HZ: f32 = 18_000.0;
const UNDERWATER_LOWPASS_FAR_HZ: f32 = 220.0;
const BARRIER_LOWPASS_NEAR_HZ: f32 = 18_000.0;
const BARRIER_LOWPASS_FAR_HZ: f32 = 900.0;
// Resonance on the shared underwater SFX filter (see `underwater_sfx_filter`) -
// a self-resonant low-pass gives a "wow"/warped underwater character on top of
// the muffling itself, rather than just sounding quieter and duller. Kept
// moderate (kira clamps to 0.0-1.0, ~1.0 is near self-oscillation) so it reads
// as natural rather than a phaser effect.
const SFX_UNDERWATER_FILTER_RESONANCE: f32 = 0.55;
// How quickly the shared underwater filter's cutoff snaps to a newly spawned
// voice's own depth-based cutoff - short enough to feel immediate for a single
// division, smooth enough not to click.
const SFX_UNDERWATER_FILTER_TWEEN: Duration = Duration::from_millis(50);

// -- World echo (cave walls / world sphere boundary) -------------------------
// Artistic in-game "speed of sound" - not literal m/s (these world units aren't
// meters), tuned so echoes at this world's scale land in a perceptible range.
const ENV_SOUND_SPEED: f32 = 400.0;
const ENV_ECHO_MIN_DISTANCE: f32 = 0.5;
const ENV_ECHO_MAX_DISTANCE: f32 = 240.0;
const ENV_ECHO_MIN_DELAY: Duration = Duration::from_millis(30);
const ENV_ECHO_MAX_DELAY: Duration = Duration::from_millis(900);
// Time constant for gliding the echo bus toward its target - roughly how long
// it takes to close ~63% of the gap after the nearest material changes. Short
// enough to feel responsive, long enough that walking past a rock/sand
// boundary crossfades rather than flips.
const WORLD_ECHO_SMOOTH_TIME_CONSTANT: f32 = 0.45;
// Reflectivity + brightness per material, loosely physically motivated: hard
// smooth surfaces (rock, glass) reflect strongly and keep their highs; soft
// porous surfaces (sand) absorb most of the energy and dull what bounces back.
const ROCK_REFLECTIVITY: f32 = 1.4;
const ROCK_ECHO_LOWPASS_HZ: f32 = 4_500.0;
const ROCK_ECHO_RESONANCE: f32 = 0.22;
const GLASS_REFLECTIVITY: f32 = 1.7;
const GLASS_ECHO_LOWPASS_HZ: f32 = 15_000.0;
const GLASS_ECHO_RESONANCE: f32 = 0.38;
const SAND_REFLECTIVITY: f32 = 0.65;
const SAND_ECHO_LOWPASS_HZ: f32 = 1_000.0;
const SAND_ECHO_RESONANCE: f32 = 0.04;
// How often the echo send track's live parameters (volume/cutoff/delay time)
// get pushed to their handles. Every frame would work too but this is plenty
// smooth for something that's already being lerped, and cheaper.
const WORLD_ECHO_PARAM_TWEEN: Duration = Duration::from_millis(120);

// Master boundary gate for "camera outside the world sphere" - see
// `AudioLayer::set_world_boundary_distance`. Inside the world this stays at
// unity right up to the boundary; outside, all audio falls off quickly with
// distance beyond the sphere so leaving/re-entering feels spatial instead of
// hard-muted.
const WORLD_BOUNDARY_FADE_DISTANCE: f32 = 200.0;
const WORLD_BOUNDARY_VOLUME_TWEEN: Duration = Duration::from_millis(45);

// -- Ambient environment drone -------------------------------------------------
// Looping CC0 recordings (AMBIENT_WIND_PATH / AMBIENT_UNDERWATER_PATH).
const DRONE_FADE_DURATION: Duration = Duration::from_millis(1000);
// Comfortably longer than DRONE_FADE_DURATION so a new crossfade never starts
// before the previous one has actually finished and settled.
const DRONE_MIN_REGEN_INTERVAL: Duration = Duration::from_millis(2500);
const DRONE_AIR_BASE_VOLUME: f32 = 0.72;
const DRONE_AIR_LOWPASS_HZ: f32 = 7_000.0;
const DRONE_WATER_BASE_VOLUME: f32 = 1.04;
const DRONE_WATER_LOWPASS_HZ: f32 = 350.0;
// Continuous, cheap proximity reactivity: multiplies the drone's volume based
// on `world_echo.amplitude` (nearest-wall closeness x material reflectivity -
// already smoothed every frame for the echo tap, reused here for free).
const DRONE_PROXIMITY_VOLUME_GAIN: f32 = 1.2;
const DRONE_VOLUME_TWEEN: Duration = Duration::from_millis(150);

const DEFAULT_MUSIC_VOLUME: f32 = 0.18;
const DEFAULT_SFX_VOLUME: f32 = 0.45;
/// Flat attenuation applied to both volume sliders' raw 0.0-1.0 value before
/// it's used anywhere. Playtesting at a "balanced" middle slider setting
/// (both sliders around 0.5) came out too loud across the board - rather
/// than lower the slider *defaults* (which wouldn't help anyone who already
/// has settings saved, and doesn't fix the mapping for other slider
/// positions either), this scales the whole 0-1 range down so the same
/// slider position everyone is already used to now produces a quieter,
/// better-calibrated result without them having to touch anything.
const USER_VOLUME_SCALE: f32 = 2.0 / 3.0;
const MENU_HOVER_VOLUME: f32 = 1.25;
const MENU_SELECT_VOLUME: f32 = 1.55;
const SLIDER_TICK_VOLUME: f32 = 0.9;
const SLIDER_TICK_MIN_INTERVAL: Duration = Duration::from_millis(83);
const CELL_MODE_SELECT_VOLUME: f32 = 1.15;
const MUSIC_FADE_DURATION: Duration = Duration::from_millis(1400);
const MAIN_MENU_MUSIC_PATH: &str = "assets/music/tracks/bio_spheres_main_menu_remaster_v0_1.wav";
const PREVIEW_MUSIC_PATH: &str = "assets/music/tracks/h_project_9_preview.mp3";
// CC0 - see assets/ambience/CREDITS.md.
const AMBIENT_WIND_PATH: &str = "assets/ambience/wind_ambience_dhallcomposer_cc0.mp3";
const AMBIENT_UNDERWATER_PATH: &str = "assets/ambience/underwater_ambience_tim_verberne_cc0.mp3";
const THERMAL_VENT_AIR_PATH: &str =
    "assets/sfx/environment/thermal_vents/vent_steam_continuous_0new4y_cc0.ogg";
const THERMAL_VENT_MAX_VOICES: usize = 64;
const THERMAL_VENT_AIR_FULL_VOLUME_RADIUS: f32 = 28.0;
const THERMAL_VENT_AIR_AUDIBLE_RADIUS: f32 = 140.0;
const THERMAL_VENT_AIR_FAR_GAIN: f32 = 0.18;
const THERMAL_VENT_AIR_SILENCE_RADIUS: f32 = 260.0;
const THERMAL_VENT_UNDERWATER_FULL_VOLUME_RADIUS: f32 = 60.0;
const THERMAL_VENT_UNDERWATER_AUDIBLE_RADIUS: f32 = 180.0;
const THERMAL_VENT_UNDERWATER_FAR_GAIN: f32 = 0.0;
const THERMAL_VENT_UNDERWATER_SILENCE_RADIUS: f32 = 360.0;
const THERMAL_VENT_AIR_BASE_VOLUME: f32 = 2.8;
const THERMAL_VENT_UNDERWATER_DRONE_GAIN: f32 = 4.0;
const THERMAL_VENT_VOLUME_TWEEN: Duration = Duration::from_millis(220);
const FLOWING_WATER_PATH: &str =
    "assets/sfx/environment/water/creek_06_loop_vkproduktion_cc0_preview.ogg";
const RAIN_LOOP_PATH: &str = "assets/sfx/environment/rain/rain_loopable_ylmir_cc0.ogg";
// Generous headroom above what any real scene is expected to need
// simultaneously - kept well above the GPU-side MAX_FLOW_SOURCES/
// MAX_RAIN_SOURCES caps (water_audio_sources_from_buckets in gpu_simulator.rs)
// so this truncation is a safety ceiling, not an active cutoff. A tight cap
// here used to mean sources at the "Nth nearest" boundary would spawn/stop as
// the camera moved past it, instead of smoothly crossfading via distance
// falloff like every other source - every source that actually reaches this
// layer now gets its own persistent voice, so proximity alone (continuously,
// via `update_environmental_loop_voice_group`) decides how audible it is.
const FLOWING_WATER_MAX_VOICES: usize = 64;
const RAIN_MAX_VOICES: usize = 40;
const FLOWING_WATER_FULL_VOLUME_RADIUS: f32 = 60.0;
const FLOWING_WATER_AUDIBLE_RADIUS: f32 = 260.0;
const FLOWING_WATER_FAR_GAIN: f32 = 0.14;
const FLOWING_WATER_SILENCE_RADIUS: f32 = 500.0;
const RAIN_FULL_VOLUME_RADIUS: f32 = 120.0;
const RAIN_AUDIBLE_RADIUS: f32 = 420.0;
const RAIN_FAR_GAIN: f32 = 0.28;
const RAIN_SILENCE_RADIUS: f32 = 700.0;
const FLOWING_WATER_BASE_VOLUME: f32 = 0.55;
const RAIN_BASE_VOLUME: f32 = 0.5;
const WATER_AMBIENCE_VOLUME_TWEEN: Duration = Duration::from_millis(280);
// Significant, deliberately obvious down-pitch for flowing water/rain loops
// while the listener is underwater - multiplies each voice's own base pitch
// (see `EnvironmentalLoopVoice::base_pitch`) rather than replacing it, so the
// natural per-voice variation is preserved, just transposed down.
const UNDERWATER_AMBIENCE_PITCH_MULTIPLIER: f32 = 0.5;
const BUTTON_HOVER_BYTES: &[u8] =
    include_bytes!("../assets/sfx/processed/button_click/h_click4_hover.mp3");
const BUTTON_SELECT_BYTES: &[u8] =
    include_bytes!("../assets/sfx/processed/button_click/h_click6_select.mp3");
const SLIDER_TICK_BYTES: &[u8] =
    include_bytes!("../assets/sfx/processed/button_click/h_click7_slider.mp3");
const CELL_MODE_SELECT_BYTES: &[u8] =
    include_bytes!("../assets/sfx/processed/button_click/h_pop1_mode_select.mp3");
const CELL_DIVIDE_BYTES: &[u8] = include_bytes!(
    "../assets/sfx/processed/cell_divide/v0_6/dry/cell_divide_slime_membrane_tear_wet_v0_6.wav"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicTrack {
    MainMenu,
    Preview,
}

impl MusicTrack {
    fn path(self) -> &'static str {
        match self {
            MusicTrack::MainMenu => MAIN_MENU_MUSIC_PATH,
            MusicTrack::Preview => PREVIEW_MUSIC_PATH,
        }
    }
}

/// Gameplay event that should produce runtime audio.
#[derive(Debug, Clone, Copy)]
pub enum GameAudioEvent {
    CellDivide {
        position: Vec3,
        burst_count: usize,
        environment: AudioEnvironment,
    },
    SliderTick,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AudioEnvironment {
    pub emitter_underwater: bool,
    pub listener_underwater: bool,
}

impl AudioEnvironment {
    pub const fn air() -> Self {
        Self {
            emitter_underwater: false,
            listener_underwater: false,
        }
    }

    fn with_listener_underwater(mut self, listener_underwater: bool) -> Self {
        self.listener_underwater = listener_underwater;
        self
    }
}

struct MusicPlayer {
    track: MusicTrack,
    handle: StaticSoundHandle,
}

/// Which kind of surface is nearest the listener, for material-dependent echo
/// and ambient tone. Rock and Sand come from the cave SDF raymarch; Glass is
/// the outer world sphere boundary, which is always present whether or not
/// caves exist in this world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentSurface {
    Rock,
    Sand,
    Glass,
}

/// A single reflected-echo tap driven by the nearest wall's distance and
/// material. `set_world_environment` sets a *target* `WorldEcho` the instant
/// the nearest surface changes; `AudioLayer` continuously glides its actual
/// (playback-facing) `WorldEcho` toward that target every frame via `lerp`,
/// so walking past a rock/sand/glass boundary crossfades the echo's tone
/// instead of snapping it.
#[derive(Debug, Clone, Copy)]
struct WorldEcho {
    delay: Duration,
    /// 0.0 = no audible echo.
    amplitude: f32,
    lowpass_hz: f32,
    /// Kira filter resonance on the echo send, used as cheap post-reflection
    /// material coloration: glass rings, rock has body, sand stays absorbed.
    resonance: f32,
}

impl WorldEcho {
    const NONE: Self = Self {
        delay: Duration::ZERO,
        amplitude: 0.0,
        lowpass_hz: 20_000.0,
        resonance: 0.0,
    };

    fn from_hit(surface: EnvironmentSurface, distance: f32) -> Self {
        if !(ENV_ECHO_MIN_DISTANCE..=ENV_ECHO_MAX_DISTANCE).contains(&distance) {
            return Self::NONE;
        }

        let (reflectivity, lowpass_hz, resonance) = match surface {
            EnvironmentSurface::Rock => {
                (ROCK_REFLECTIVITY, ROCK_ECHO_LOWPASS_HZ, ROCK_ECHO_RESONANCE)
            }
            EnvironmentSurface::Glass => (
                GLASS_REFLECTIVITY,
                GLASS_ECHO_LOWPASS_HZ,
                GLASS_ECHO_RESONANCE,
            ),
            EnvironmentSurface::Sand => {
                (SAND_REFLECTIVITY, SAND_ECHO_LOWPASS_HZ, SAND_ECHO_RESONANCE)
            }
        };

        // Round-trip time at an artistic in-game "speed of sound" - not literal
        // m/s, tuned so echoes at this world's cave scale (tens of units) land
        // in a perceptible 30-900ms range instead of being physically instant.
        let delay_secs = (2.0 * distance / ENV_SOUND_SPEED).clamp(
            ENV_ECHO_MIN_DELAY.as_secs_f32(),
            ENV_ECHO_MAX_DELAY.as_secs_f32(),
        );
        let t = (distance / ENV_ECHO_MAX_DISTANCE).clamp(0.0, 1.0);
        let closeness = (1.0 - t).powf(1.5);

        Self {
            delay: Duration::from_secs_f32(delay_secs),
            amplitude: reflectivity * closeness,
            lowpass_hz,
            resonance,
        }
    }

    /// Moves `self` a fraction `t` of the way toward `target`. `t` is derived
    /// from elapsed time by the caller so the glide rate is frame-rate
    /// independent.
    fn lerp(self, target: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let delay_secs =
            self.delay.as_secs_f32() + (target.delay.as_secs_f32() - self.delay.as_secs_f32()) * t;
        Self {
            delay: Duration::from_secs_f32(delay_secs.max(0.0)),
            amplitude: self.amplitude + (target.amplitude - self.amplitude) * t,
            lowpass_hz: self.lowpass_hz + (target.lowpass_hz - self.lowpass_hz) * t,
            resonance: self.resonance + (target.resonance - self.resonance) * t,
        }
    }
}

/// Which base loop is loaded - the one dimension that genuinely requires
/// swapping the underlying sound, so it stays on the debounced/cooldown path.
/// Everything else (how loud/present it is right now) is continuous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DroneParams {
    underwater: bool,
}

impl DroneParams {
    fn base_volume(self) -> f32 {
        if self.underwater {
            DRONE_WATER_BASE_VOLUME
        } else {
            DRONE_AIR_BASE_VOLUME
        }
    }

    fn lowpass_hz(self) -> f32 {
        if self.underwater {
            DRONE_WATER_LOWPASS_HZ
        } else {
            DRONE_AIR_LOWPASS_HZ
        }
    }

    fn path(self) -> &'static str {
        if self.underwater {
            AMBIENT_UNDERWATER_PATH
        } else {
            AMBIENT_WIND_PATH
        }
    }
}

struct AmbientDrone {
    params: DroneParams,
    handle: StaticSoundHandle,
}

#[derive(Debug)]
struct ThermalVentVoice {
    position: Vec3,
    handle: StaticSoundHandle,
}

#[derive(Debug, Clone, Copy)]
pub struct EnvironmentalAudioSource {
    pub position: Vec3,
    pub strength: f32,
}

#[derive(Debug)]
struct EnvironmentalLoopVoice {
    position: Vec3,
    strength: f32,
    /// Playback rate this voice was spawned with (natural per-voice variation,
    /// see `spawn_environmental_loop_voices`) - kept so the continuous
    /// underwater pitch multiplier in `update_environmental_loop_voice_group`
    /// can be applied on top of it each frame instead of overwriting it.
    base_pitch: f32,
    handle: StaticSoundHandle,
}

/// Small playback parameter bundle for one SFX instance.
#[derive(Debug, Clone, Copy)]
struct SfxParams {
    /// Playback speed multiplier - changes pitch and duration together,
    /// which is exactly what we want for organic variation.
    pitch: f32,
    /// Linear gain multiplier.
    volume: f32,
    /// Emitter position in world space.
    position: Vec3,
}

impl SfxParams {
    fn cell_divide(rng: &mut TinyRng, burst_scale: f32, position: Vec3) -> Self {
        Self {
            pitch: cell_divide_pitch_from_rng(rng),
            volume: rng.range_f32(0.75, 0.95) * burst_scale,
            position,
        }
    }
}

/// Listener transform used for manual distance/pan math. Kira does have a
/// native spatial-track system, but it's built around few, persistent,
/// positioned tracks (e.g. one per game entity) - not hundreds of one-shot
/// bursts at unique positions fired every frame. This reuses the distance
/// falloff and stereo panning math already proven out, converting the result
/// to kira's Decibels/Panning at the point of playback instead.
#[derive(Debug, Clone, Copy)]
struct AudioListener {
    position: Vec3,
    rotation: Quat,
}

impl AudioListener {
    fn from_camera(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    /// -1.0 (full left) .. 1.0 (full right), from the emitter's direction
    /// relative to listener facing. Distance doesn't affect pan, only volume.
    fn pan(self, world_position: Vec3) -> f32 {
        let relative = world_position - self.position;
        if relative.length_squared() < 1e-6 {
            return 0.0;
        }
        let local = self.rotation.inverse() * relative;
        let horizontal = Vec3::new(local.x, 0.0, local.z);
        if horizontal.length_squared() < 1e-6 {
            return 0.0;
        }
        // Angle from forward (-Z, matching this app's camera convention),
        // sine gives a natural full-left/full-right mapping at +/-90 degrees.
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let right = Vec3::new(1.0, 0.0, 0.0);
        let dir = horizontal.normalize();
        let angle = dir.dot(forward).clamp(-1.0, 1.0).acos() * dir.dot(right).signum();
        (angle.sin()).clamp(-1.0, 1.0)
    }

    fn distance_sq(self, world_position: Vec3) -> f32 {
        world_position.distance_squared(self.position)
    }

    fn cell_divide_distance_gain(self, world_position: Vec3) -> f32 {
        let distance = self.distance_sq(world_position).sqrt();
        distance_falloff(
            distance,
            CELL_DIVIDE_FULL_VOLUME_RADIUS,
            CELL_DIVIDE_AUDIBLE_RADIUS,
            CELL_DIVIDE_FAR_GAIN,
            CELL_DIVIDE_SILENCE_RADIUS,
        )
    }
}

/// Two-stage distance falloff: full volume out to `full_volume_radius`, a
/// smoothstep taper down to `far_gain` at `far_radius` (still audible, just
/// quiet - gives ambient awareness of distant activity), then a second
/// smoothstep taper from `far_gain` down to true 0.0 at `silence_radius`.
/// Without the second stage, everything past `far_radius` sits at the same
/// `far_gain` floor forever, no matter how far away it actually is.
fn distance_falloff(
    distance: f32,
    full_volume_radius: f32,
    far_radius: f32,
    far_gain: f32,
    silence_radius: f32,
) -> f32 {
    if distance <= full_volume_radius {
        return 1.0;
    }
    if distance >= silence_radius {
        return 0.0;
    }
    if distance <= far_radius {
        let t = ((distance - full_volume_radius) / (far_radius - full_volume_radius).max(0.001))
            .clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        1.0 + (far_gain - 1.0) * smooth
    } else {
        let t =
            ((distance - far_radius) / (silence_radius - far_radius).max(0.001)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        far_gain * (1.0 - smooth)
    }
}

fn thermal_vent_distance_gain(distance: f32, underwater: bool) -> f32 {
    if underwater {
        distance_falloff(
            distance,
            THERMAL_VENT_UNDERWATER_FULL_VOLUME_RADIUS,
            THERMAL_VENT_UNDERWATER_AUDIBLE_RADIUS,
            THERMAL_VENT_UNDERWATER_FAR_GAIN,
            THERMAL_VENT_UNDERWATER_SILENCE_RADIUS,
        )
    } else {
        distance_falloff(
            distance,
            THERMAL_VENT_AIR_FULL_VOLUME_RADIUS,
            THERMAL_VENT_AIR_AUDIBLE_RADIUS,
            THERMAL_VENT_AIR_FAR_GAIN,
            THERMAL_VENT_AIR_SILENCE_RADIUS,
        )
    }
}

fn world_boundary_gain(signed_distance_inside: f32) -> f32 {
    if signed_distance_inside >= 0.0 {
        return 1.0;
    }

    let outside_distance = -signed_distance_inside;
    let t = (outside_distance / WORLD_BOUNDARY_FADE_DISTANCE).clamp(0.0, 1.0);
    1.0 - t
}

fn thermal_vent_hash(position: Vec3, index: usize) -> u32 {
    let qx = (position.x * 10.0).round() as i32 as u32;
    let qy = (position.y * 10.0).round() as i32 as u32;
    let qz = (position.z * 10.0).round() as i32 as u32;
    let mut h = 0xA17E_511Du32 ^ (index as u32).wrapping_mul(0x9E37_79B9);
    h ^= qx.wrapping_mul(0x85EB_CA6B);
    h = h.rotate_left(7) ^ qy.wrapping_mul(0xC2B2_AE35);
    h = h.rotate_left(11) ^ qz.wrapping_mul(0x27D4_EB2D);
    h ^= h >> 15;
    h.wrapping_mul(0x2C1B_3C6D) ^ (h >> 12)
}

fn thermal_vent_offset_secs(position: Vec3, index: usize, duration_secs: f64) -> f64 {
    let unit = thermal_vent_hash(position, index) as f64 / u32::MAX as f64;
    unit * duration_secs
}

fn thermal_vent_pitch(position: Vec3, index: usize) -> f32 {
    let unit = thermal_vent_hash(position, index ^ 0x517) as f32 / u32::MAX as f32;
    0.82 + unit * 0.12
}

fn update_environmental_loop_voice_group(
    listener: &AudioListener,
    voices: &mut Vec<EnvironmentalLoopVoice>,
    base_volume: f32,
    full_volume_radius: f32,
    audible_radius: f32,
    far_gain: f32,
    silence_radius: f32,
    group_gain: f32,
    pitch_multiplier: f32,
) {
    if voices.is_empty() {
        return;
    }

    let voice_count = voices.len().max(1) as f32;
    let base_volume = base_volume / voice_count.sqrt();
    let tween = Tween {
        duration: WATER_AMBIENCE_VOLUME_TWEEN,
        easing: Easing::Linear,
        ..Default::default()
    };

    for voice in voices.iter_mut() {
        let distance = listener.distance_sq(voice.position).sqrt();
        let distance_gain = distance_falloff(
            distance,
            full_volume_radius,
            audible_radius,
            far_gain,
            silence_radius,
        );
        let target_volume = base_volume * voice.strength * group_gain * distance_gain;
        voice
            .handle
            .set_volume(db_from_linear(target_volume), tween);
        voice
            .handle
            .set_panning(Panning(listener.pan(voice.position)), tween);
        voice.handle.set_playback_rate(
            PlaybackRate((voice.base_pitch * pitch_multiplier) as f64),
            tween,
        );
    }

    voices.retain(|voice| voice.handle.state() != PlaybackState::Stopped);
}

/// One in-flight cell-divide voice, tracked only so the pool can evict the
/// farthest one when full (kira's `StaticSoundHandle` is a lightweight `Copy`
/// reference, not an RAII guard - dropping it does *not* stop playback, so
/// nothing needs to be "kept alive" for a voice to play out normally).
struct ActiveSfx {
    handle: StaticSoundHandle,
    position: Vec3,
    distance_sq: f32,
}

#[derive(Debug, Clone, Copy)]
struct PendingCellDivisions {
    position: Vec3,
    distance_sq: f32,
    environment: AudioEnvironment,
}

#[derive(Debug, Clone, Copy)]
struct CellDivideAcoustics {
    pitch: f32,
    volume_gain: f32,
    /// Low-pass cutoff in Hz, or `None` for no filtering (plain air, close range).
    lowpass_cutoff_hz: Option<f32>,
}

/// Resident audio service owned by `App`.
pub struct AudioLayer {
    manager: Option<AudioManager<DefaultBackend>>,
    /// Dedicated bus all plain-air cell-divide voices play through - fixed
    /// capacity of MAX_SFX_VOICES, and sends a portion of its output to the
    /// shared echo bus (built in `build_echo_send`) so every voice
    /// contributes to one continuously-modulated echo instead of each
    /// carrying its own filter.
    sfx_track: Option<TrackHandle>,
    /// Sister bus to `sfx_track` for cell-divide voices with any underwater
    /// muffling (either side of the barrier underwater, see
    /// `cell_divide_acoustics`) - carries one persistent, resonant low-pass
    /// filter (`underwater_sfx_filter`) shared by every voice currently
    /// routed through it, snapped to each new voice's own depth-based cutoff
    /// in `spawn_cell_divide_voice`. Same "no per-voice DSP" cost model as
    /// the echo bus: a handful of bus-level filters regardless of how many
    /// voices are alive, giving a genuinely filtered, slightly warped tone
    /// instead of a volume trim standing in for one.
    underwater_sfx_track: Option<TrackHandle>,
    underwater_sfx_filter: Option<FilterHandle>,
    /// Live handles into the shared echo bus's effect chain (Delay -> Filter
    /// -> VolumeControl, see `build_echo_send`). Only Filter and VolumeControl
    /// are pushed the smoothed `world_echo` state each frame - the delay time
    /// itself is fixed at construction, kira's DelayHandle only exposes
    /// changing feedback/mix live, not delay_time.
    echo_filter: Option<FilterHandle>,
    echo_volume: Option<VolumeControlHandle>,
    /// Dedicated bus for the ambient drone, with its own Filter for air/water
    /// tone (materials don't get individual per-voice filters - a continuous
    /// background loop can afford exactly one). The filter is created once
    /// with the track and reused across drone regenerations by calling
    /// `set_cutoff` on it - kira's tracks don't support adding effects after
    /// the track itself is built.
    ambient_track: Option<TrackHandle>,
    ambient_filter: Option<FilterHandle>,

    music: Option<MusicPlayer>,
    music_volume: f32,
    sfx_volume: f32,
    button_hover: Vec<u8>,
    button_select: Vec<u8>,
    slider_tick: Vec<u8>,
    cell_mode_select: Vec<u8>,
    cell_divide: Vec<u8>,
    last_slider_tick_at: Option<Instant>,
    active_sfx: VecDeque<ActiveSfx>,
    pending_cell_divisions: Vec<PendingCellDivisions>,
    pending_cell_divisions_first_seen_at: Option<Instant>,
    listener: AudioListener,
    listener_underwater: bool,
    rng: TinyRng,
    /// Token-bucket rate limit on how fast *new* SFX voices can be created,
    /// independent of MAX_SFX_VOICES (how many can be simultaneously active).
    voice_create_tokens: f32,
    voice_create_tokens_updated_at: Instant,
    /// Playback-facing echo tap, continuously smoothed toward `world_echo_target`.
    world_echo: WorldEcho,
    /// Where `world_echo` is gliding toward, set instantly by `set_world_environment`.
    world_echo_target: WorldEcho,
    world_echo_smoothed_at: Instant,
    ambient_drone: Option<AmbientDrone>,
    thermal_vent_voices: Vec<ThermalVentVoice>,
    thermal_vent_sources: Vec<Vec3>,
    thermal_vent_underwater: bool,
    thermal_vent_air_loop: Option<StaticSoundData>,
    flowing_water_voices: Vec<EnvironmentalLoopVoice>,
    flowing_water_sources: Vec<EnvironmentalAudioSource>,
    rain_voices: Vec<EnvironmentalLoopVoice>,
    rain_sources: Vec<EnvironmentalAudioSource>,
    rain_intensity: f32,
    flowing_water_loop: Option<StaticSoundData>,
    rain_loop: Option<StaticSoundData>,
    /// Parameters the current `ambient_drone` was generated from, so repeated
    /// `set_world_environment` calls with a near-identical environment don't
    /// keep tearing the drone down and crossfading for no audible reason.
    ambient_drone_target_params: Option<DroneParams>,
    /// When the drone last actually regenerated. A moving listener's distance
    /// to the nearest wall changes on nearly every throttled sample, so
    /// comparing params alone isn't enough of a debounce - without a minimum
    /// interval, a new crossfade could start before the previous one
    /// finished repeatedly.
    ambient_drone_last_regen_at: Option<Instant>,
    /// Last main-track gain set by the world boundary gate, so repeated calls
    /// with effectively the same state don't re-issue volume commands.
    world_boundary_gain: f32,
}

impl AudioLayer {
    pub fn new() -> Self {
        Self::new_with_volumes(DEFAULT_MUSIC_VOLUME, DEFAULT_SFX_VOLUME)
    }

    pub fn new_with_volumes(music_volume: f32, sfx_volume: f32) -> Self {
        let mut manager = match AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
        {
            Ok(manager) => Some(manager),
            Err(err) => {
                log::warn!("Audio disabled: failed to open default output device: {err}");
                None
            }
        };

        let (echo_send, echo_filter, echo_volume) = manager
            .as_mut()
            .and_then(|manager| build_echo_send(manager))
            .map(|(send, filter, volume)| (Some(send), Some(filter), Some(volume)))
            .unwrap_or((None, None, None));

        let sfx_track = manager.as_mut().and_then(|manager| {
            let mut builder = TrackBuilder::new().sound_capacity(MAX_SFX_VOICES);
            if let Some(echo_send) = &echo_send {
                builder = builder.with_send(echo_send, Decibels(6.0));
            }
            manager.add_sub_track(builder).ok()
        });

        let mut underwater_sfx_builder = TrackBuilder::new().sound_capacity(MAX_SFX_VOICES);
        if let Some(echo_send) = &echo_send {
            underwater_sfx_builder = underwater_sfx_builder.with_send(echo_send, Decibels(6.0));
        }
        let underwater_sfx_filter_handle = underwater_sfx_builder.add_effect(
            FilterBuilder::new()
                .cutoff(UNDERWATER_LOWPASS_NEAR_HZ as f64)
                .resonance(SFX_UNDERWATER_FILTER_RESONANCE as f64),
        );
        let underwater_sfx_track = manager
            .as_mut()
            .and_then(|manager| manager.add_sub_track(underwater_sfx_builder).ok());
        let underwater_sfx_filter = underwater_sfx_track
            .as_ref()
            .map(|_| underwater_sfx_filter_handle);

        let mut ambient_builder = TrackBuilder::new();
        let ambient_filter_handle =
            ambient_builder.add_effect(FilterBuilder::new().cutoff(DRONE_AIR_LOWPASS_HZ as f64));
        let ambient_track = manager
            .as_mut()
            .and_then(|manager| manager.add_sub_track(ambient_builder).ok());
        let ambient_filter = ambient_track.as_ref().map(|_| ambient_filter_handle);

        let mut audio = Self {
            manager,
            sfx_track,
            underwater_sfx_track,
            underwater_sfx_filter,
            echo_filter,
            echo_volume,
            ambient_track,
            ambient_filter,
            music: None,
            music_volume: music_volume.clamp(0.0, 1.0) * USER_VOLUME_SCALE,
            sfx_volume: sfx_volume.clamp(0.0, 1.0) * USER_VOLUME_SCALE,
            button_hover: BUTTON_HOVER_BYTES.to_vec(),
            button_select: BUTTON_SELECT_BYTES.to_vec(),
            slider_tick: SLIDER_TICK_BYTES.to_vec(),
            cell_mode_select: CELL_MODE_SELECT_BYTES.to_vec(),
            cell_divide: CELL_DIVIDE_BYTES.to_vec(),
            last_slider_tick_at: None,
            active_sfx: VecDeque::with_capacity(MAX_SFX_VOICES),
            pending_cell_divisions: Vec::with_capacity(MAX_PENDING_CELL_DIVISION_EVENTS),
            pending_cell_divisions_first_seen_at: None,
            listener: AudioListener::from_camera(Vec3::ZERO, Quat::IDENTITY),
            listener_underwater: false,
            rng: TinyRng::new(0xB105_FEEE_D1A1_DEAD),
            voice_create_tokens: MAX_SFX_VOICES as f32,
            voice_create_tokens_updated_at: Instant::now(),
            world_echo: WorldEcho::NONE,
            world_echo_target: WorldEcho::NONE,
            world_echo_smoothed_at: Instant::now(),
            ambient_drone: None,
            thermal_vent_voices: Vec::new(),
            thermal_vent_sources: Vec::new(),
            thermal_vent_underwater: false,
            thermal_vent_air_loop: load_static_loop(THERMAL_VENT_AIR_PATH),
            flowing_water_voices: Vec::new(),
            flowing_water_sources: Vec::new(),
            rain_voices: Vec::new(),
            rain_sources: Vec::new(),
            rain_intensity: 0.0,
            flowing_water_loop: load_static_loop(FLOWING_WATER_PATH),
            rain_loop: load_static_loop(RAIN_LOOP_PATH),
            ambient_drone_target_params: None,
            ambient_drone_last_regen_at: None,
            world_boundary_gain: 1.0,
        };
        audio.set_music_track(Some(MusicTrack::MainMenu));
        audio
    }

    pub fn set_volumes(&mut self, music_volume: f32, sfx_volume: f32) {
        self.music_volume = music_volume.clamp(0.0, 1.0) * USER_VOLUME_SCALE;
        self.sfx_volume = sfx_volume.clamp(0.0, 1.0) * USER_VOLUME_SCALE;
        if let Some(music) = &mut self.music {
            music
                .handle
                .set_volume(db_from_linear(self.music_volume), instant_tween());
        }
    }

    /// Keep this in sync with the active scene camera before playing spatial SFX.
    pub fn set_listener_from_camera(&mut self, position: Vec3, rotation: Quat) {
        self.listener = AudioListener::from_camera(position, rotation);
    }

    pub fn set_listener_environment(&mut self, underwater: bool) {
        self.listener_underwater = underwater;
    }

    /// Refreshes the reflected-echo tap and ambient drone from the nearest
    /// wall/boundary (if any) and whether the listener is underwater. Cheap
    /// to call but not free (may spin up a crossfade) - the caller throttles
    /// this rather than recomputing every frame.
    pub fn set_world_environment(
        &mut self,
        wall: Option<(EnvironmentSurface, f32)>,
        underwater: bool,
        thermal_vent_sources: &[Vec3],
        flowing_water_sources: &[EnvironmentalAudioSource],
        rain_sources: &[EnvironmentalAudioSource],
        rain_intensity: f32,
    ) {
        self.world_echo_target = match wall {
            Some((surface, distance)) => WorldEcho::from_hit(surface, distance),
            None => WorldEcho::NONE,
        };
        self.set_ambient_drone_params(DroneParams { underwater });
        self.set_thermal_vent_sources(thermal_vent_sources, underwater);
        self.set_environmental_loop_sources(flowing_water_sources, rain_sources, rain_intensity);
    }

    pub fn set_water_environment(
        &mut self,
        flowing_water_sources: &[EnvironmentalAudioSource],
        rain_sources: &[EnvironmentalAudioSource],
        rain_intensity: f32,
    ) {
        self.set_environmental_loop_sources(flowing_water_sources, rain_sources, rain_intensity);
    }

    /// Silences the world echo and fades out the ambient drone - used when
    /// there's no meaningful world geometry to react to (e.g. outside GPU
    /// scene mode). The echo glides to silence like any other target change
    /// rather than cutting out immediately.
    pub fn clear_world_environment(&mut self) {
        self.world_echo_target = WorldEcho::NONE;
        if self.ambient_drone_target_params.take().is_some() {
            self.ambient_drone_last_regen_at = None;
            if let Some(mut drone) = self.ambient_drone.take() {
                drone.handle.stop(fade_tween(DRONE_FADE_DURATION));
            }
        }
        self.stop_thermal_vents();
        self.stop_environmental_loop_voices();
    }

    /// Master boundary gate for all audio routed through Kira's main track.
    /// `signed_distance_inside` is positive inside the sphere and negative
    /// outside. Approaching the boundary from inside does not attenuate at all;
    /// only distance beyond the boundary is converted into a steep falloff.
    pub fn set_world_boundary_distance(&mut self, signed_distance_inside: Option<f32>) {
        let target_gain = signed_distance_inside.map_or(1.0, world_boundary_gain);
        if (self.world_boundary_gain - target_gain).abs() < 0.01 {
            return;
        }
        self.world_boundary_gain = target_gain;

        let Some(manager) = &mut self.manager else {
            return;
        };
        manager.main_track().set_volume(
            db_from_linear(target_gain),
            Tween {
                duration: WORLD_BOUNDARY_VOLUME_TWEEN,
                easing: Easing::Linear,
                ..Default::default()
            },
        );
    }

    fn set_ambient_drone_params(&mut self, params: DroneParams) {
        if self.ambient_drone_target_params == Some(params) {
            return;
        }
        if let Some(last_regen) = self.ambient_drone_last_regen_at {
            if last_regen.elapsed() < DRONE_MIN_REGEN_INTERVAL {
                return;
            }
        }
        self.ambient_drone_last_regen_at = Some(Instant::now());
        self.ambient_drone_target_params = Some(params);

        if self.manager.is_none() {
            return;
        }
        let Some(ambient_track) = &mut self.ambient_track else {
            return;
        };

        if let Some(mut old) = self.ambient_drone.take() {
            old.handle.stop(fade_tween(DRONE_FADE_DURATION));
        }

        let path = asset_path(params.path());
        let Ok(data) = StaticSoundData::from_file(&path) else {
            log::warn!("Ambient drone disabled: could not load {}", path.display());
            return;
        };
        let data = data
            .loop_region(..)
            .volume(Decibels::SILENCE)
            .fade_in_tween(Some(fade_tween(DRONE_FADE_DURATION)));

        let Ok(mut handle) = ambient_track.play(data) else {
            log::warn!("Ambient drone disabled: failed to start playback");
            return;
        };
        if let Some(filter) = &mut self.ambient_filter {
            filter.set_cutoff(params.lowpass_hz() as f64, fade_tween(DRONE_FADE_DURATION));
        }
        handle.set_volume(
            db_from_linear(params.base_volume() * self.sfx_volume),
            fade_tween(DRONE_FADE_DURATION),
        );

        self.ambient_drone = Some(AmbientDrone { params, handle });
    }

    fn update_ambient_drone(&mut self) {
        let Some(drone) = &mut self.ambient_drone else {
            return;
        };
        if drone.handle.state() == PlaybackState::Stopped {
            self.ambient_drone = None;
            return;
        }

        // Continuous and cheap: reuses world_echo.amplitude, which
        // update_world_echo_smoothing already glides every frame for the
        // echo tap, so the drone's presence tracks material proximity in
        // real time without ever needing to reload or refilter the loop.
        let vent_resonance_gain = if drone.params.underwater {
            self.thermal_vent_sources
                .iter()
                .map(|source| {
                    let distance = self.listener.distance_sq(*source).sqrt();
                    thermal_vent_distance_gain(distance, true)
                })
                .fold(0.0_f32, f32::max)
                * THERMAL_VENT_UNDERWATER_DRONE_GAIN
        } else {
            0.0
        };
        let proximity_gain =
            1.0 + self.world_echo.amplitude * DRONE_PROXIMITY_VOLUME_GAIN + vent_resonance_gain;
        let target =
            (drone.params.base_volume() * proximity_gain * self.sfx_volume).clamp(0.0, 3.0);
        drone.handle.set_volume(
            db_from_linear(target),
            Tween {
                duration: DRONE_VOLUME_TWEEN,
                ..Default::default()
            },
        );
    }

    fn set_thermal_vent_sources(&mut self, sources: &[Vec3], underwater: bool) {
        let mut sources: Vec<Vec3> = sources
            .iter()
            .copied()
            .take(THERMAL_VENT_MAX_VOICES)
            .collect();
        sources.sort_by(|a, b| {
            a.x.total_cmp(&b.x)
                .then(a.y.total_cmp(&b.y))
                .then(a.z.total_cmp(&b.z))
        });

        let same_sources = self.thermal_vent_sources.len() == sources.len()
            && self
                .thermal_vent_sources
                .iter()
                .zip(&sources)
                .all(|(a, b)| a.distance_squared(*b) < 0.01);
        if same_sources && self.thermal_vent_underwater == underwater {
            return;
        }

        self.stop_thermal_vent_voices();
        self.thermal_vent_sources = sources;
        self.thermal_vent_underwater = underwater;

        if underwater || self.manager.is_none() || self.thermal_vent_sources.is_empty() {
            return;
        }

        let base_loop = self.thermal_vent_air_loop.clone();
        let Some(base_loop) = base_loop else {
            return;
        };

        let duration_secs = base_loop.duration().as_secs_f64().max(0.001);
        let Some(manager) = &mut self.manager else {
            return;
        };

        for (index, &position) in self.thermal_vent_sources.iter().enumerate() {
            let offset = thermal_vent_offset_secs(position, index, duration_secs);
            let pitch = thermal_vent_pitch(position, index);
            let data = base_loop
                .loop_region(..)
                .start_position(offset)
                .volume(Decibels::SILENCE)
                .playback_rate(PlaybackRate(pitch as f64))
                .panning(Panning(self.listener.pan(position)))
                .fade_in_tween(Some(fade_tween(DRONE_FADE_DURATION)));

            let played = self
                .sfx_track
                .as_mut()
                .and_then(|track| track.play(data.clone()).ok())
                .or_else(|| manager.play(data).ok());
            if let Some(handle) = played {
                self.thermal_vent_voices
                    .push(ThermalVentVoice { position, handle });
            }
        }
    }

    fn stop_thermal_vent_voices(&mut self) {
        for mut voice in self.thermal_vent_voices.drain(..) {
            voice.handle.stop(fade_tween(DRONE_FADE_DURATION));
        }
    }

    fn stop_thermal_vents(&mut self) {
        self.stop_thermal_vent_voices();
        self.thermal_vent_sources.clear();
    }

    fn update_thermal_vents(&mut self) {
        if self.thermal_vent_voices.is_empty() {
            return;
        }

        let listener = self.listener;
        let voice_count = self.thermal_vent_voices.len().max(1) as f32;
        let base_volume = THERMAL_VENT_AIR_BASE_VOLUME;
        let mix_compensation = (1.0 / voice_count.sqrt()).clamp(0.35, 1.0);
        let tween = Tween {
            duration: THERMAL_VENT_VOLUME_TWEEN,
            ..Default::default()
        };

        for voice in &mut self.thermal_vent_voices {
            if voice.handle.state() == PlaybackState::Stopped {
                continue;
            }
            let distance = listener.distance_sq(voice.position).sqrt();
            let distance_gain = thermal_vent_distance_gain(distance, false);
            let target = base_volume * mix_compensation * distance_gain * self.sfx_volume;
            voice.handle.set_volume(db_from_linear(target), tween);
            voice
                .handle
                .set_panning(Panning(listener.pan(voice.position)), tween);
        }

        self.thermal_vent_voices
            .retain(|voice| voice.handle.state() != PlaybackState::Stopped);
    }

    fn set_environmental_loop_sources(
        &mut self,
        flowing_water_sources: &[EnvironmentalAudioSource],
        rain_sources: &[EnvironmentalAudioSource],
        rain_intensity: f32,
    ) {
        let mut flow = flowing_water_sources.to_vec();
        flow.sort_by(|a, b| {
            self.listener
                .distance_sq(a.position)
                .total_cmp(&self.listener.distance_sq(b.position))
        });
        flow.truncate(FLOWING_WATER_MAX_VOICES);

        let mut rain = rain_sources.to_vec();
        let rain_intensity = rain_intensity.clamp(0.0, 1.0);
        if rain_intensity > 0.0 {
            rain.push(EnvironmentalAudioSource {
                position: self.listener.position,
                strength: 1.0,
            });
        }
        rain.sort_by(|a, b| {
            self.listener
                .distance_sq(a.position)
                .total_cmp(&self.listener.distance_sq(b.position))
        });
        rain.truncate(RAIN_MAX_VOICES);

        let flow_loop = self.flowing_water_loop.clone();
        let mut flow_voices = std::mem::take(&mut self.flowing_water_voices);
        self.reconcile_environmental_loop_voices(&mut flow_voices, &flow, flow_loop, 0xA611);
        self.flowing_water_voices = flow_voices;
        self.flowing_water_sources = flow;

        if rain_intensity > 0.0 {
            let rain_loop = self.rain_loop.clone();
            let mut rain_voices = std::mem::take(&mut self.rain_voices);
            self.reconcile_environmental_loop_voices(&mut rain_voices, &rain, rain_loop, 0xB411);
            self.rain_voices = rain_voices;
        } else {
            for mut voice in self.rain_voices.drain(..) {
                voice.handle.stop(fade_tween(WATER_AMBIENCE_VOLUME_TWEEN));
            }
        }
        self.rain_sources = rain;
        self.rain_intensity = rain_intensity;
    }

    /// Matches existing voices to current sources by position rather than
    /// list index. `sources` is freshly re-sorted on every call - by
    /// distance-to-listener here, and by a strength value that fluctuates
    /// every physics step on the GPU extraction side - so index-based
    /// matching (the old approach) treated harmless reordering as every
    /// source disappearing and reappearing, tearing every voice down and
    /// rebuilding it before its fade-in ever finished. Position-based
    /// matching only actually spawns or stops a voice when a source
    /// genuinely appears or disappears.
    fn reconcile_environmental_loop_voices(
        &mut self,
        voices: &mut Vec<EnvironmentalLoopVoice>,
        sources: &[EnvironmentalAudioSource],
        base_loop: Option<StaticSoundData>,
        salt: usize,
    ) {
        let mut matched = vec![false; sources.len()];
        voices.retain_mut(|voice| {
            let closest = sources
                .iter()
                .enumerate()
                .filter(|(i, _)| !matched[*i])
                .min_by(|(_, a), (_, b)| {
                    voice
                        .position
                        .distance_squared(a.position)
                        .total_cmp(&voice.position.distance_squared(b.position))
                });
            match closest {
                Some((idx, source))
                    if voice.position.distance_squared(source.position) < 4096.0 =>
                {
                    matched[idx] = true;
                    voice.position = source.position;
                    voice.strength = source.strength.clamp(0.0, 1.0);
                    true
                }
                _ => {
                    voice.handle.stop(fade_tween(WATER_AMBIENCE_VOLUME_TWEEN));
                    false
                }
            }
        });

        let Some(base_loop) = base_loop else {
            return;
        };
        let new_sources: Vec<EnvironmentalAudioSource> = sources
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched[*i])
            .map(|(_, source)| *source)
            .collect();
        if new_sources.is_empty() {
            return;
        }
        let spawned = self.spawn_environmental_loop_voices(base_loop, &new_sources, salt);
        voices.extend(spawned);
    }

    fn spawn_environmental_loop_voices(
        &mut self,
        base_loop: StaticSoundData,
        sources: &[EnvironmentalAudioSource],
        salt: usize,
    ) -> Vec<EnvironmentalLoopVoice> {
        let Some(manager) = &mut self.manager else {
            return Vec::new();
        };
        let duration_secs = base_loop.duration().as_secs_f64().max(0.001);
        let mut voices = Vec::with_capacity(sources.len());
        for (index, source) in sources.iter().enumerate() {
            let offset = thermal_vent_offset_secs(source.position, index ^ salt, duration_secs);
            let pitch = 0.96
                + (thermal_vent_hash(source.position, index ^ salt) as f32 / u32::MAX as f32)
                    * 0.08;
            let data = base_loop
                .clone()
                .loop_region(..)
                .start_position(offset)
                .volume(Decibels::SILENCE)
                .playback_rate(PlaybackRate(pitch as f64))
                .panning(Panning(self.listener.pan(source.position)))
                .fade_in_tween(Some(fade_tween(WATER_AMBIENCE_VOLUME_TWEEN)));
            let handle = self
                .sfx_track
                .as_mut()
                .and_then(|track| track.play(data.clone()).ok())
                .or_else(|| manager.play(data).ok());
            if let Some(handle) = handle {
                voices.push(EnvironmentalLoopVoice {
                    position: source.position,
                    strength: source.strength.clamp(0.0, 1.0),
                    base_pitch: pitch,
                    handle,
                });
            }
        }
        voices
    }

    fn stop_environmental_loop_voices(&mut self) {
        for mut voice in self.flowing_water_voices.drain(..) {
            voice.handle.stop(fade_tween(WATER_AMBIENCE_VOLUME_TWEEN));
        }
        for mut voice in self.rain_voices.drain(..) {
            voice.handle.stop(fade_tween(WATER_AMBIENCE_VOLUME_TWEEN));
        }
        self.flowing_water_sources.clear();
        self.rain_sources.clear();
        self.rain_intensity = 0.0;
    }

    fn update_environmental_loops(&mut self) {
        let pitch_multiplier = if self.listener_underwater {
            UNDERWATER_AMBIENCE_PITCH_MULTIPLIER
        } else {
            1.0
        };
        update_environmental_loop_voice_group(
            &self.listener,
            &mut self.flowing_water_voices,
            FLOWING_WATER_BASE_VOLUME,
            FLOWING_WATER_FULL_VOLUME_RADIUS,
            FLOWING_WATER_AUDIBLE_RADIUS,
            FLOWING_WATER_FAR_GAIN,
            FLOWING_WATER_SILENCE_RADIUS,
            1.0,
            pitch_multiplier,
        );
        update_environmental_loop_voice_group(
            &self.listener,
            &mut self.rain_voices,
            RAIN_BASE_VOLUME,
            RAIN_FULL_VOLUME_RADIUS,
            RAIN_AUDIBLE_RADIUS,
            RAIN_FAR_GAIN,
            RAIN_SILENCE_RADIUS,
            self.rain_intensity,
            pitch_multiplier,
        );
    }

    pub fn play_event(&mut self, event: GameAudioEvent) {
        match event {
            GameAudioEvent::CellDivide {
                position,
                burst_count,
                environment,
            } => self.queue_cell_divisions(burst_count, position, environment),
            GameAudioEvent::SliderTick => self.play_slider_tick(),
        }
    }

    pub fn update(&mut self) {
        self.prune_finished();
        self.flush_cell_division_queue();
        self.update_music_fades();
        self.update_ambient_drone();
        self.update_thermal_vents();
        self.update_environmental_loops();
        self.update_world_echo();
    }

    /// Glides the playback-facing `world_echo` toward `world_echo_target`
    /// every frame, independent of how often `set_world_environment` itself
    /// is called (the caller throttles that; this runs every frame so the
    /// glide stays smooth regardless), and pushes the result to the shared
    /// echo bus's effect handles. The delay's own time-per-repeat is fixed at
    /// construction (kira's DelayHandle doesn't expose changing it live, only
    /// feedback/mix) - tone (Filter cutoff) and audibility (VolumeControl)
    /// still track distance/material continuously, which carries most of the
    /// perceptible "is there an echo, and what does it sound like" effect.
    fn update_world_echo(&mut self) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.world_echo_smoothed_at)
            .as_secs_f32();
        self.world_echo_smoothed_at = now;

        let alpha = 1.0 - (-dt / WORLD_ECHO_SMOOTH_TIME_CONSTANT).exp();
        self.world_echo = self.world_echo.lerp(self.world_echo_target, alpha);

        let tween = Tween {
            duration: WORLD_ECHO_PARAM_TWEEN,
            ..Default::default()
        };
        let echo = self.world_echo;
        if let Some(filter) = &mut self.echo_filter {
            filter.set_cutoff(echo.lowpass_hz as f64, tween);
            filter.set_resonance(echo.resonance as f64, tween);
        }
        if let Some(volume) = &mut self.echo_volume {
            volume.set_volume(db_from_linear(echo.amplitude), tween);
        }
    }

    pub fn play_menu_hover(&mut self) {
        self.play_ui_click(&self.button_hover.clone(), MENU_HOVER_VOLUME);
    }

    pub fn play_menu_select(&mut self) {
        self.play_ui_click(&self.button_select.clone(), MENU_SELECT_VOLUME);
    }

    pub fn play_slider_tick(&mut self) {
        let now = Instant::now();
        if let Some(last_tick_at) = self.last_slider_tick_at {
            if now.duration_since(last_tick_at) < SLIDER_TICK_MIN_INTERVAL {
                return;
            }
        }
        self.last_slider_tick_at = Some(now);
        self.play_ui_click(&self.slider_tick.clone(), SLIDER_TICK_VOLUME);
    }

    pub fn play_cell_mode_select(&mut self) {
        self.play_ui_click(&self.cell_mode_select.clone(), CELL_MODE_SELECT_VOLUME);
    }

    pub fn play_cell_divide_burst_scaled(&mut self, divisions_this_frame: usize) {
        self.play_cell_divide_burst_scaled_at(divisions_this_frame, Vec3::ZERO);
    }

    /// Play one spatialized cell division sound, scaling volume down when many
    /// divisions happen in the same frame.
    pub fn play_cell_divide_burst_scaled_at(
        &mut self,
        divisions_this_frame: usize,
        position: Vec3,
    ) {
        let environment =
            AudioEnvironment::air().with_listener_underwater(self.listener_underwater);
        let distance_sq = self.listener.distance_sq(position);
        if !self.reserve_sfx_slot(distance_sq) {
            return;
        }

        let burst_scale = match divisions_this_frame {
            0..=8 => 1.0,
            9..=32 => 0.82,
            33..=96 => 0.62,
            _ => 0.45,
        };
        let params = SfxParams::cell_divide(&mut self.rng, burst_scale, position);
        let acoustics =
            cell_divide_acoustics(self.listener, environment, params.position, params.pitch);
        let volume = params.volume * acoustics.volume_gain * self.sfx_volume;
        let pan = self.listener.pan(position);

        if let Some(handle) =
            self.spawn_cell_divide_voice(acoustics.pitch, volume, pan, acoustics.lowpass_cutoff_hz)
        {
            self.active_sfx.push_back(ActiveSfx {
                handle,
                position,
                distance_sq,
            });
        }
    }

    pub fn active_sfx_count(&mut self) -> usize {
        self.prune_finished();
        self.active_sfx.len()
    }

    pub fn set_music_track(&mut self, target: Option<MusicTrack>) {
        if self.music.as_ref().map(|music| music.track) == target {
            return;
        }

        if let Some(mut old_music) = self.music.take() {
            old_music.handle.stop(fade_tween(MUSIC_FADE_DURATION));
        }

        let Some(track) = target else {
            return;
        };
        let Some(manager) = &mut self.manager else {
            return;
        };

        let path = asset_path(track.path());
        let Ok(data) = StaticSoundData::from_file(&path) else {
            log::warn!("Music disabled: could not load {}", path.display());
            return;
        };
        let data = data
            .loop_region(..)
            .volume(Decibels::SILENCE)
            .fade_in_tween(Some(fade_tween(MUSIC_FADE_DURATION)));

        let Ok(mut handle) = manager.play(data) else {
            log::warn!("Music disabled: failed to start playback");
            return;
        };
        handle.set_volume(
            db_from_linear(self.music_volume),
            fade_tween(MUSIC_FADE_DURATION),
        );

        self.music = Some(MusicPlayer { track, handle });
    }

    pub fn start_main_menu_music_loop(&mut self) {
        self.set_music_track(Some(MusicTrack::MainMenu));
    }

    /// Music doesn't need per-frame bookkeeping the way the drone does (no
    /// continuous reactivity, just an occasional swap) - kira's own fade
    /// tweens already handle the in/out transition. This just drops the
    /// handle once a fade-out has actually finished.
    fn update_music_fades(&mut self) {
        if let Some(music) = &self.music {
            if music.handle.state() == PlaybackState::Stopped {
                self.music = None;
            }
        }
    }

    fn play_ui_click(&mut self, bytes: &[u8], volume: f32) {
        let Some(manager) = &mut self.manager else {
            return;
        };
        let Ok(data) = StaticSoundData::from_cursor(Cursor::new(bytes.to_vec())) else {
            log::warn!("Failed to decode UI click SFX");
            return;
        };
        let _ = manager.play(data.volume(db_from_linear(volume * self.sfx_volume)));
    }

    fn queue_cell_divisions(
        &mut self,
        count: usize,
        position: Vec3,
        environment: AudioEnvironment,
    ) {
        if count == 0 {
            return;
        }

        let now = Instant::now();
        if self.pending_cell_divisions_first_seen_at.is_none() {
            self.pending_cell_divisions_first_seen_at = Some(now);
        }

        for _ in 0..count {
            self.queue_one_cell_division(position, environment);
        }
    }

    fn queue_one_cell_division(&mut self, position: Vec3, environment: AudioEnvironment) {
        let environment = environment
            .with_listener_underwater(environment.listener_underwater || self.listener_underwater);
        let pending = PendingCellDivisions {
            position,
            distance_sq: self.listener.distance_sq(position),
            environment,
        };

        if self.pending_cell_divisions.len() >= MAX_PENDING_CELL_DIVISION_EVENTS {
            return;
        }

        self.pending_cell_divisions.push(pending);
    }

    fn flush_cell_division_queue(&mut self) {
        let Some(first_seen_at) = self.pending_cell_divisions_first_seen_at else {
            return;
        };

        let now = Instant::now();
        if now.duration_since(first_seen_at) < CELL_DIVIDE_COALESCE_WINDOW {
            return;
        }

        let pending = std::mem::take(&mut self.pending_cell_divisions);
        self.pending_cell_divisions_first_seen_at = None;
        self.play_cell_divide_cluster(&pending);
    }

    /// Play the nearest newly arrived divisions, letting fresh split events
    /// replace older tails when the voice pool is full.
    fn play_cell_divide_cluster(&mut self, pending: &[PendingCellDivisions]) {
        self.prune_finished();
        if pending.is_empty() || self.manager.is_none() {
            return;
        }

        let listener = self.listener;
        for active in self.active_sfx.iter_mut() {
            active.distance_sq = listener.distance_sq(active.position);
        }

        let mut candidates = pending.to_vec();
        for candidate in &mut candidates {
            candidate.distance_sq = listener.distance_sq(candidate.position);
        }

        let target_count = candidates.len().min(MAX_SFX_VOICES);
        if target_count == 0 {
            return;
        }

        if candidates.len() > target_count {
            candidates.select_nth_unstable_by(target_count, |a, b| {
                a.distance_sq.total_cmp(&b.distance_sq)
            });
            candidates.truncate(target_count);
        }

        // Letting per-voice volume keep decaying with cluster size (instead of
        // flooring it) keeps total burst loudness roughly constant regardless
        // of how many divisions land in one coalesce window - a floor here
        // meant a 150-voice burst summed many times louder than a 20-voice
        // one instead of holding steady, which could drown out everything
        // else including the ambient drone.
        let per_event_volume = (1.65 / (target_count as f32).sqrt()).clamp(0.05, 0.85);

        for candidate in candidates.iter() {
            if !self.reserve_sfx_slot(candidate.distance_sq) {
                continue;
            }

            let pitch = cell_divide_pitch_from_rng(&mut self.rng);
            let acoustics =
                cell_divide_acoustics(listener, candidate.environment, candidate.position, pitch);
            let volume = per_event_volume * acoustics.volume_gain * self.sfx_volume;
            let pan = listener.pan(candidate.position);

            if let Some(handle) = self.spawn_cell_divide_voice(
                acoustics.pitch,
                volume,
                pan,
                acoustics.lowpass_cutoff_hz,
            ) {
                self.active_sfx.push_back(ActiveSfx {
                    handle,
                    position: candidate.position,
                    distance_sq: candidate.distance_sq,
                });
            }
        }
    }

    /// Builds and plays one cell-divide voice. Plain-air voices (no
    /// muffling) play dry on `sfx_track`; anything with underwater muffling
    /// plays on `underwater_sfx_track` instead, whose shared resonant filter
    /// is snapped to this event's own depth-based cutoff first - genuinely
    /// filtered and slightly warped rather than just quieter. Both busses
    /// send a fixed proportion of their output to the shared echo bus (see
    /// `echo_send`), so neither ever needs a per-voice filter/echo chain,
    /// unlike the old rodio implementation.
    fn spawn_cell_divide_voice(
        &mut self,
        pitch: f32,
        volume: f32,
        pan: f32,
        lowpass_cutoff_hz: Option<f32>,
    ) -> Option<StaticSoundHandle> {
        let cursor = Cursor::new(self.cell_divide.clone());
        let data = StaticSoundData::from_cursor(cursor).ok()?;
        let data = data
            .volume(db_from_linear(volume))
            .playback_rate(PlaybackRate(pitch as f64))
            .panning(Panning(pan));

        if let Some(cutoff_hz) = lowpass_cutoff_hz {
            if let Some(filter) = &mut self.underwater_sfx_filter {
                filter.set_cutoff(cutoff_hz as f64, fade_tween(SFX_UNDERWATER_FILTER_TWEEN));
            }
            let underwater_sfx_track = self.underwater_sfx_track.as_mut()?;
            return underwater_sfx_track.play(data).ok();
        }

        let sfx_track = self.sfx_track.as_mut()?;
        sfx_track.play(data).ok()
    }

    fn prune_finished(&mut self) {
        self.active_sfx
            .retain(|active| active.handle.state() != PlaybackState::Stopped);
    }

    /// Token-bucket check: true (and consumes a token) if creating a new
    /// voice right now respects MAX_VOICE_CREATES_PER_SEC, false if not.
    /// Bucket capacity equals MAX_SFX_VOICES, so a burst can still fill the
    /// whole pool instantly after a quiet period - only *sustained* creation
    /// is throttled, not how many can exist at once.
    fn try_consume_voice_create_token(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now
            .duration_since(self.voice_create_tokens_updated_at)
            .as_secs_f32();
        self.voice_create_tokens_updated_at = now;
        self.voice_create_tokens = (self.voice_create_tokens + elapsed * MAX_VOICE_CREATES_PER_SEC)
            .min(MAX_SFX_VOICES as f32);

        if self.voice_create_tokens >= 1.0 {
            self.voice_create_tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn reserve_sfx_slot(&mut self, distance_sq: f32) -> bool {
        if !self.try_consume_voice_create_token() {
            return false;
        }
        self.prune_finished();
        if self.active_sfx.len() < MAX_SFX_VOICES {
            return true;
        }

        let listener = self.listener;
        for active in &mut self.active_sfx {
            active.distance_sq = listener.distance_sq(active.position);
        }

        let Some((farthest_idx, farthest)) = self
            .active_sfx
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.distance_sq.total_cmp(&b.distance_sq))
        else {
            return true;
        };

        if distance_sq > farthest.distance_sq {
            return false;
        }

        if let Some(mut evicted) = self.active_sfx.remove(farthest_idx) {
            evicted.handle.stop(instant_tween());
        }
        true
    }
}

impl Default for AudioLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the shared echo send bus: Delay (the actual repeats) -> Filter
/// (material tone) -> VolumeControl (how audible it is right now, driven by
/// `world_echo.amplitude`). One instance shared by every echoing voice - see
/// the `echo_filter`/`echo_volume` field docs on `AudioLayer`.
fn build_echo_send(
    manager: &mut AudioManager<DefaultBackend>,
) -> Option<(SendTrackHandle, FilterHandle, VolumeControlHandle)> {
    // Delay time is fixed at construction - kira's DelayHandle can't change it
    // live (only feedback/mix), so use an obvious slapback rather than the old
    // midpoint of the 30-900ms range, which read more like a late stray sound.
    let fixed_delay_time = Duration::from_millis(185);
    let mut builder = SendTrackBuilder::new();
    builder.add_effect(
        DelayBuilder::new()
            .delay_time(fixed_delay_time)
            .feedback(Decibels(-2.0))
            .mix(Mix::WET),
    );
    let filter = builder.add_effect(FilterBuilder::new().cutoff(20_000.0));
    let volume = builder.add_effect(VolumeControlBuilder::new(Decibels::SILENCE));
    let send = manager.add_send_track(builder).ok()?;
    Some((send, filter, volume))
}

fn asset_path(relative_path: impl AsRef<Path>) -> PathBuf {
    let relative_path = relative_path.as_ref();
    let cwd_path = PathBuf::from(relative_path);
    if cwd_path.exists() {
        return cwd_path;
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative_path)
}

fn load_static_loop(relative_path: &str) -> Option<StaticSoundData> {
    let path = asset_path(relative_path);
    match StaticSoundData::from_file(&path) {
        Ok(data) => Some(data.loop_region(..)),
        Err(err) => {
            log::warn!(
                "Audio loop disabled: could not load {}: {err}",
                path.display()
            );
            None
        }
    }
}

/// `Decibels::from(f32)` (kira's own `From<f32>` impl) treats its input as an
/// already-in-decibels value, NOT a linear amplitude - it does no log
/// conversion. Every volume in this file (distance falloff, echo amplitude,
/// drone proximity gain, UI SFX volumes) is computed as a 0.0-1.0-ish linear
/// gain, so it has to go through an actual amplitude->dB conversion here
/// instead of kira's `From` impl, or all of those volumes collapse to a
/// barely-audible +/-1dB range around unity gain regardless of distance.
fn db_from_linear(linear: f32) -> Decibels {
    let linear = linear.max(0.0);
    if linear <= 0.0001 {
        return Decibels::SILENCE;
    }
    Decibels(20.0 * linear.log10())
}

fn instant_tween() -> Tween {
    Tween {
        duration: Duration::from_millis(15),
        easing: Easing::Linear,
        ..Default::default()
    }
}

fn fade_tween(duration: Duration) -> Tween {
    Tween {
        duration,
        ..Default::default()
    }
}

/// Low-pass cutoff for sound traveling through water, interpolated smoothly from
/// `near_hz` (effectively unfiltered, close to the source) down to `far_hz` (only
/// a low thump survives) as distance approaches `attenuation_radius`. This is the
/// actual physical mechanism behind "underwater sounds are muffled": absorption in
/// water rises sharply with frequency, so high frequencies scatter/absorb within a
/// short range while low frequencies keep traveling - not a flat volume penalty.
fn water_lowpass_cutoff_hz(
    distance: f32,
    full_volume_radius: f32,
    attenuation_radius: f32,
    near_hz: f32,
    far_hz: f32,
) -> f32 {
    let t = ((distance - full_volume_radius)
        / (attenuation_radius - full_volume_radius).max(0.001))
    .clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    near_hz + (far_hz - near_hz) * smooth
}

fn cell_divide_acoustics(
    listener: AudioListener,
    environment: AudioEnvironment,
    position: Vec3,
    base_pitch: f32,
) -> CellDivideAcoustics {
    let emitter_underwater = environment.emitter_underwater;
    let listener_underwater = environment.listener_underwater;
    let distance = listener.distance_sq(position).sqrt();

    if emitter_underwater && listener_underwater {
        let underwater_distance_gain = distance_falloff(
            distance,
            CELL_DIVIDE_FULL_VOLUME_RADIUS,
            UNDERWATER_CELL_DIVIDE_AUDIBLE_RADIUS,
            UNDERWATER_CELL_DIVIDE_FAR_GAIN,
            UNDERWATER_CELL_DIVIDE_SILENCE_RADIUS,
        );
        let near_pressure = if distance <= CELL_DIVIDE_FULL_VOLUME_RADIUS {
            0.78
        } else {
            1.0
        };
        let cutoff_hz = water_lowpass_cutoff_hz(
            distance,
            CELL_DIVIDE_FULL_VOLUME_RADIUS,
            UNDERWATER_CELL_DIVIDE_SILENCE_RADIUS,
            UNDERWATER_LOWPASS_NEAR_HZ,
            UNDERWATER_LOWPASS_FAR_HZ,
        );

        return CellDivideAcoustics {
            pitch: base_pitch * 0.64,
            volume_gain: underwater_distance_gain * near_pressure,
            lowpass_cutoff_hz: Some(cutoff_hz),
        };
    }

    if emitter_underwater || listener_underwater {
        let air_gain = listener.cell_divide_distance_gain(position);
        let cutoff_hz = water_lowpass_cutoff_hz(
            distance,
            CELL_DIVIDE_FULL_VOLUME_RADIUS,
            CELL_DIVIDE_AUDIBLE_RADIUS,
            BARRIER_LOWPASS_NEAR_HZ,
            BARRIER_LOWPASS_FAR_HZ,
        );
        return CellDivideAcoustics {
            pitch: base_pitch * 0.82,
            volume_gain: air_gain * 0.42,
            lowpass_cutoff_hz: Some(cutoff_hz),
        };
    }

    CellDivideAcoustics {
        pitch: base_pitch,
        volume_gain: listener.cell_divide_distance_gain(position),
        lowpass_cutoff_hz: None,
    }
}

#[derive(Debug, Clone, Copy)]
struct TinyRng {
    state: u64,
}

impl TinyRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        let unit = self.next_u32() as f32 / u32::MAX as f32;
        min + (max - min) * unit
    }

    fn range_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }

        self.next_u32() as usize % max
    }
}

fn cell_divide_pitch_from_rng(rng: &mut TinyRng) -> f32 {
    const SEMITONES: [i32; 9] = [-5, -3, -2, 0, 2, 3, 5, 7, 9];
    let index = rng.range_usize(SEMITONES.len());
    let detune = rng.range_f32(-0.018, 0.018);
    2.0_f32.powf(SEMITONES[index] as f32 / 12.0) + detune
}
