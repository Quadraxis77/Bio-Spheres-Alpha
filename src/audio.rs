//! Lightweight runtime audio for music and short game SFX.
//!
//! The layer is intentionally small: it streams music from assets, preloads game
//! SFX bytes, mixes through rodio, and caps concurrent voices so bursty
//! simulation events stay cheap.

use glam::{Quat, Vec3};
use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player, Source, SpatialPlayer};
use std::{
    collections::VecDeque,
    fs::File,
    io::Cursor,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const MAX_SFX_VOICES: usize = 256;
const CELL_DIVIDE_DURATION: Duration = Duration::from_millis(380);
const CELL_DIVIDE_COALESCE_WINDOW: Duration = Duration::from_millis(55);
const MAX_PENDING_CELL_DIVISION_EVENTS: usize = 250_000;
const AUDIO_SPACE_RADIUS: f32 = 24.0;
// Rodio's `Spatial` source applies its own inverse-square distance falloff
// between the emitter and each ear (clamped to at most 1.0, i.e. no boost).
// That falloff must stay fully clamped for every mapped emitter position, or
// it silently stacks with `cell_divide_distance_gain` below and crushes
// volume far below what the game's own distance curve intends. With ears at
// `EAR_HALF_WIDTH` apart, the worst-case emitter-to-ear distance is
// `AUDIO_SPACE_EXTENT + EAR_HALF_WIDTH`, which must stay under 1.0 for the
// clamp to hold everywhere - so this only ever drives stereo panning.
const AUDIO_SPACE_EXTENT: f32 = 0.7;
const CELL_DIVIDE_FULL_VOLUME_RADIUS: f32 = 14.0;
const CELL_DIVIDE_AUDIBLE_RADIUS: f32 = 70.0;
const CELL_DIVIDE_FAR_GAIN: f32 = 0.08;
const EAR_HALF_WIDTH: f32 = 0.22;
const DEFAULT_MUSIC_VOLUME: f32 = 0.18;
const DEFAULT_SFX_VOLUME: f32 = 0.45;
const MENU_HOVER_VOLUME: f32 = 0.22;
const MENU_SELECT_VOLUME: f32 = 0.28;
const MENU_SELECT_PITCH: f32 = 1.06;
const MAIN_MENU_MUSIC_PATH: &str =
    "assets/music/tracks/bio_spheres_main_menu_remaster_v0_1.wav";
const BUTTON_CLICK_BYTES: &[u8] =
    include_bytes!("../assets/sfx/processed/button_click/button_click_bio_ui_v0_3_tight_no_thump.wav");
const CELL_DIVIDE_BYTES: &[u8] = include_bytes!(
    "../assets/sfx/processed/cell_divide/v0_6/dry/cell_divide_slime_membrane_tear_wet_v0_6.wav"
);

/// Gameplay event that should produce runtime audio.
#[derive(Debug, Clone, Copy)]
pub enum GameAudioEvent {
    CellDivide {
        position: Vec3,
        burst_count: usize,
    },
}

/// Small playback parameter bundle for one SFX instance.
#[derive(Debug, Clone, Copy)]
pub struct SfxParams {
    /// Playback speed multiplier. Rodio's `speed` changes pitch and duration together,
    /// which is exactly what we want for organic variation.
    pub pitch: f32,
    /// Linear gain multiplier.
    pub volume: f32,
    /// Emitter position in world space.
    pub position: Vec3,
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

/// Listener transform used by the spatial mixer.
#[derive(Debug, Clone, Copy)]
pub struct AudioListener {
    pub position: Vec3,
    pub rotation: Quat,
}

impl AudioListener {
    pub fn from_camera(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    fn ears(self) -> ([f32; 3], [f32; 3]) {
        let left = Vec3::new(-EAR_HALF_WIDTH, 0.0, 0.0);
        let right = Vec3::new(EAR_HALF_WIDTH, 0.0, 0.0);
        (left.to_array(), right.to_array())
    }

    fn emitter_position(self, world_position: Vec3) -> [f32; 3] {
        let relative = world_position - self.position;
        let local = self.rotation.inverse() * relative;
        let distance = local.length();
        let mapped = if distance > AUDIO_SPACE_RADIUS {
            local.normalize_or_zero() * AUDIO_SPACE_EXTENT
        } else {
            local * (AUDIO_SPACE_EXTENT / AUDIO_SPACE_RADIUS)
        };
        mapped.to_array()
    }

    fn distance_sq(self, world_position: Vec3) -> f32 {
        world_position.distance_squared(self.position)
    }

    fn cell_divide_distance_gain(self, world_position: Vec3) -> f32 {
        let distance = self.distance_sq(world_position).sqrt();
        if distance <= CELL_DIVIDE_FULL_VOLUME_RADIUS {
            return 1.0;
        }

        let t = ((distance - CELL_DIVIDE_FULL_VOLUME_RADIUS)
            / (CELL_DIVIDE_AUDIBLE_RADIUS - CELL_DIVIDE_FULL_VOLUME_RADIUS))
            .clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        1.0 + (CELL_DIVIDE_FAR_GAIN - 1.0) * smooth
    }
}

struct ActiveSfx {
    expires_at: Instant,
    position: Vec3,
    distance_sq: f32,
    _player: SpatialPlayer,
}

#[derive(Debug, Clone, Copy)]
struct PendingCellDivisions {
    position: Vec3,
    distance_sq: f32,
}

/// Resident audio service owned by `App`.
pub struct AudioLayer {
    sink: Option<MixerDeviceSink>,
    music: Option<Player>,
    music_volume: f32,
    sfx_volume: f32,
    button_click: Vec<u8>,
    cell_divide: Vec<u8>,
    active_sfx: VecDeque<ActiveSfx>,
    pending_cell_divisions: Vec<PendingCellDivisions>,
    pending_cell_divisions_first_seen_at: Option<Instant>,
    listener: AudioListener,
    rng: TinyRng,
}

impl AudioLayer {
    pub fn new() -> Self {
        Self::new_with_volumes(DEFAULT_MUSIC_VOLUME, DEFAULT_SFX_VOLUME)
    }

    pub fn new_with_volumes(music_volume: f32, sfx_volume: f32) -> Self {
        let sink = match DeviceSinkBuilder::open_default_sink() {
            Ok(sink) => Some(sink),
            Err(err) => {
                log::warn!("Audio disabled: failed to open default output device: {err}");
                None
            }
        };

        let mut audio = Self {
            sink,
            music: None,
            music_volume: music_volume.clamp(0.0, 1.0),
            sfx_volume: sfx_volume.clamp(0.0, 1.0),
            button_click: BUTTON_CLICK_BYTES.to_vec(),
            cell_divide: CELL_DIVIDE_BYTES.to_vec(),
            active_sfx: VecDeque::with_capacity(MAX_SFX_VOICES),
            pending_cell_divisions: Vec::with_capacity(MAX_PENDING_CELL_DIVISION_EVENTS),
            pending_cell_divisions_first_seen_at: None,
            listener: AudioListener::from_camera(Vec3::ZERO, Quat::IDENTITY),
            rng: TinyRng::new(0xB105_FEEE_D1A1_DEAD),
        };
        audio.start_main_menu_music_loop();
        audio
    }

    pub fn set_volumes(&mut self, music_volume: f32, sfx_volume: f32) {
        self.music_volume = music_volume.clamp(0.0, 1.0);
        self.sfx_volume = sfx_volume.clamp(0.0, 1.0);
        if let Some(music) = &self.music {
            music.set_volume(self.music_volume);
        }
    }

    /// Keep this in sync with the active scene camera before playing spatial SFX.
    pub fn set_listener_from_camera(&mut self, position: Vec3, rotation: Quat) {
        self.listener = AudioListener::from_camera(position, rotation);
    }

    pub fn play_event(&mut self, event: GameAudioEvent) {
        match event {
            GameAudioEvent::CellDivide {
                position,
                burst_count,
            } => self.queue_cell_divisions(burst_count, position),
        }
    }

    pub fn update(&mut self) {
        self.prune_finished();
        self.flush_cell_division_queue();
    }

    pub fn play_menu_hover(&self) {
        self.play_ui_click(1.0, MENU_HOVER_VOLUME);
    }

    pub fn play_menu_select(&self) {
        self.play_ui_click(MENU_SELECT_PITCH, MENU_SELECT_VOLUME);
    }

    /// Play the accepted cell division sound with subtle per-instance variation.
    pub fn play_cell_divide(&mut self) {
        self.play_cell_divide_at(Vec3::ZERO);
    }

    /// Play one cell division sound at a world position.
    pub fn play_cell_divide_at(&mut self, position: Vec3) {
        self.play_cell_divide_burst_scaled_at(1, position);
    }

    /// Play one cell division sound, scaling volume down when many divisions
    /// happen in the same frame.
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
        let distance_sq = self.listener.distance_sq(position);
        if !self.reserve_sfx_slot(distance_sq) {
            return;
        }

        let Some(sink) = &self.sink else {
            return;
        };

        let burst_scale = match divisions_this_frame {
            0..=8 => 1.0,
            9..=32 => 0.82,
            33..=96 => 0.62,
            _ => 0.45,
        };
        let params = SfxParams::cell_divide(&mut self.rng, burst_scale, position);

        let cursor = Cursor::new(self.cell_divide.clone());
        let Ok(source) = Decoder::try_from(cursor) else {
            log::warn!("Failed to decode cell divide SFX");
            return;
        };

        let emitter = self.listener.emitter_position(params.position);
        let distance_gain = self.listener.cell_divide_distance_gain(params.position);
        let (left_ear, right_ear) = self.listener.ears();
        let player = SpatialPlayer::connect_new(sink.mixer(), emitter, left_ear, right_ear);
        player.append(
            source
                .speed(params.pitch)
                .amplify(params.volume * distance_gain * self.sfx_volume),
        );

        let lifetime = CELL_DIVIDE_DURATION.mul_f32(1.0 / params.pitch.max(0.01));
        self.active_sfx.push_back(ActiveSfx {
            expires_at: Instant::now() + lifetime,
            position,
            distance_sq,
            _player: player,
        });
    }

    pub fn active_sfx_count(&mut self) -> usize {
        self.prune_finished();
        self.active_sfx.len()
    }

    pub fn start_main_menu_music_loop(&mut self) {
        if self.music.is_some() {
            return;
        }

        let Some(sink) = &self.sink else {
            return;
        };

        let path = asset_path(MAIN_MENU_MUSIC_PATH);
        let Ok(file) = File::open(&path) else {
            log::warn!("Music disabled: could not open {}", path.display());
            return;
        };
        let Ok(source) = Decoder::try_from(file) else {
            log::warn!("Music disabled: could not decode {}", path.display());
            return;
        };

        let player = Player::connect_new(sink.mixer());
        player.set_volume(self.music_volume);
        player.append(source.repeat_infinite());
        self.music = Some(player);
    }

    fn play_ui_click(&self, pitch: f32, volume: f32) {
        let Some(sink) = &self.sink else {
            return;
        };

        let cursor = Cursor::new(self.button_click.clone());
        let Ok(source) = Decoder::try_from(cursor) else {
            log::warn!("Failed to decode menu click SFX");
            return;
        };

        sink.mixer()
            .add(source.speed(pitch).amplify(volume * self.sfx_volume));
    }

    fn queue_cell_divisions(&mut self, count: usize, position: Vec3) {
        if count == 0 {
            return;
        }

        let now = Instant::now();
        if self.pending_cell_divisions_first_seen_at.is_none() {
            self.pending_cell_divisions_first_seen_at = Some(now);
        }

        for _ in 0..count {
            self.queue_one_cell_division(position);
        }
    }

    fn queue_one_cell_division(&mut self, position: Vec3) {
        let pending = PendingCellDivisions {
            position,
            distance_sq: self.listener.distance_sq(position),
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

    /// Resync the active cell-divide voice pool so it always holds the
    /// `MAX_SFX_VOICES` closest divisions to the listener, merging newly
    /// arrived candidates with whatever is already playing. A voice that
    /// falls outside the closest set is cut immediately and its slot handed
    /// to a closer candidate, rather than being left to ring out its full
    /// lifetime and starve nearby divisions of a voice - this is what makes
    /// sound drop out in the middle of a dense, actively-dividing cluster.
    fn play_cell_divide_cluster(&mut self, pending: &[PendingCellDivisions]) {
        self.prune_finished();
        if pending.is_empty() {
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

        #[derive(Clone, Copy)]
        enum Source {
            Active(usize),
            New(usize),
        }

        let mut merged: Vec<(f32, Source)> = self
            .active_sfx
            .iter()
            .enumerate()
            .map(|(i, active)| (active.distance_sq, Source::Active(i)))
            .chain(
                candidates
                    .iter()
                    .enumerate()
                    .map(|(i, candidate)| (candidate.distance_sq, Source::New(i))),
            )
            .collect();

        let target_count = merged.len().min(MAX_SFX_VOICES);
        if merged.len() > target_count {
            merged.select_nth_unstable_by(target_count, |a, b| a.0.total_cmp(&b.0));
            merged.truncate(target_count);
        }

        let mut keep_active = vec![false; self.active_sfx.len()];
        let mut admit_new = vec![false; candidates.len()];
        for &(_, source) in &merged {
            match source {
                Source::Active(i) => keep_active[i] = true,
                Source::New(i) => admit_new[i] = true,
            }
        }

        let mut idx = 0usize;
        self.active_sfx.retain(|_| {
            let keep = keep_active[idx];
            idx += 1;
            keep
        });

        if target_count == 0 {
            return;
        }
        let per_event_volume = (1.65 / (target_count as f32).sqrt()).clamp(0.32, 0.85);

        let Some(sink) = &self.sink else {
            return;
        };

        for (i, candidate) in candidates.iter().enumerate() {
            if !admit_new[i] {
                continue;
            }

            let pitch = cell_divide_pitch_from_rng(&mut self.rng);
            let distance_gain = listener.cell_divide_distance_gain(candidate.position);
            let volume = per_event_volume * distance_gain * self.sfx_volume;

            let cursor = Cursor::new(self.cell_divide.clone());
            let Ok(source) = Decoder::try_from(cursor) else {
                log::warn!("Failed to decode cell divide SFX");
                continue;
            };

            let emitter = listener.emitter_position(candidate.position);
            let (left_ear, right_ear) = listener.ears();
            let player = SpatialPlayer::connect_new(sink.mixer(), emitter, left_ear, right_ear);
            player.append(source.speed(pitch).amplify(volume));

            let lifetime = CELL_DIVIDE_DURATION.mul_f32(1.0 / pitch.max(0.01));
            self.active_sfx.push_back(ActiveSfx {
                expires_at: Instant::now() + lifetime,
                position: candidate.position,
                distance_sq: candidate.distance_sq,
                _player: player,
            });
        }
    }

    fn prune_finished(&mut self) {
        let now = Instant::now();
        self.active_sfx.retain(|active| active.expires_at > now);
    }

    fn reserve_sfx_slot(&mut self, distance_sq: f32) -> bool {
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

        if distance_sq >= farthest.distance_sq {
            return false;
        }

        self.active_sfx.remove(farthest_idx);
        true
    }
}

fn asset_path(relative_path: impl AsRef<Path>) -> PathBuf {
    let relative_path = relative_path.as_ref();
    let cwd_path = PathBuf::from(relative_path);
    if cwd_path.exists() {
        return cwd_path;
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative_path)
}

impl Default for AudioLayer {
    fn default() -> Self {
        Self::new()
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
