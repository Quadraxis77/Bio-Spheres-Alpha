//! Lightweight toast notification system.
//!
//! Toasts appear in the bottom-right corner of the screen, fade out after a
//! few seconds, and stack vertically. Each toast has a kind (success, error,
//! info) that determines its accent color.

use egui::{Color32, Rect, Vec2};
use crate::ui::ui_system::palette;

/// How long a toast stays fully visible before fading (seconds).
const TOAST_VISIBLE_SECS: f32 = 2.5;
/// How long the fade-out takes (seconds).
const TOAST_FADE_SECS: f32 = 0.6;
/// Total lifetime = visible + fade.
const TOAST_LIFETIME: f32 = TOAST_VISIBLE_SECS + TOAST_FADE_SECS;

/// Height of one toast row (public for overlay positioning).
pub const TOAST_H_PUB: f32 = 36.0;
const TOAST_H: f32 = TOAST_H_PUB;
/// Gap between stacked toasts (public for overlay positioning).
pub const TOAST_GAP_PUB: f32 = 6.0;
const TOAST_GAP: f32 = TOAST_GAP_PUB;
/// Horizontal width of the toast.
const TOAST_W: f32 = 280.0;
/// Margin from the screen edge.
const TOAST_MARGIN: f32 = 16.0;

#[derive(Clone, Debug, PartialEq)]
pub enum ToastKind {
    Success,
    Error,
    Info,
    Progress,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Toast {
    pub message: String,
    pub kind: ToastKind,
    /// Elapsed time since this toast was created (seconds).
    pub age: f32,
    /// Optional progress 0.0-1.0 (only used for Progress kind).
    pub progress: Option<f32>,
}

impl Toast {
    pub fn success(msg: impl Into<String>) -> Self {
        Self { message: msg.into(), kind: ToastKind::Success, age: 0.0, progress: None }
    }
    pub fn error(msg: impl Into<String>) -> Self {
        Self { message: msg.into(), kind: ToastKind::Error, age: 0.0, progress: None }
    }
    pub fn info(msg: impl Into<String>) -> Self {
        Self { message: msg.into(), kind: ToastKind::Info, age: 0.0, progress: None }
    }
    pub fn progress(msg: impl Into<String>, frac: f32) -> Self {
        Self { message: msg.into(), kind: ToastKind::Progress, age: 0.0, progress: Some(frac) }
    }

    /// Alpha 0-255 based on age.
    pub fn alpha(&self) -> u8 {
        if self.age < TOAST_VISIBLE_SECS {
            255
        } else {
            let fade = ((TOAST_LIFETIME - self.age) / TOAST_FADE_SECS).clamp(0.0, 1.0);
            (fade * 255.0) as u8
        }
    }

    pub fn is_expired(&self) -> bool {
        self.age >= TOAST_LIFETIME
    }
}

/// Advance all toasts by `dt` seconds and remove expired ones.
pub fn tick_toasts(toasts: &mut Vec<Toast>, dt: f32) {
    for t in toasts.iter_mut() {
        t.age += dt;
    }
    toasts.retain(|t| !t.is_expired());
}

/// Update the progress value of the most recent Progress toast, or push a new one.
pub fn upsert_progress_toast(toasts: &mut Vec<Toast>, msg: &str, frac: f32) {
    if let Some(t) = toasts.iter_mut().rev().find(|t| t.kind == ToastKind::Progress) {
        t.message = msg.to_string();
        t.progress = Some(frac);
        t.age = 0.0; // reset timer so it stays visible
    } else {
        toasts.push(Toast::progress(msg, frac));
    }
}

/// Remove all Progress toasts (call when capture finishes).
pub fn remove_progress_toasts(toasts: &mut Vec<Toast>) {
    toasts.retain(|t| t.kind != ToastKind::Progress);
}

/// Render all active toasts in the bottom-right corner of the screen.
pub fn render_toasts(ctx: &egui::Context, toasts: &[Toast]) {
    if toasts.is_empty() { return; }

    let p = palette();
    #[allow(deprecated)]
    let screen = ctx.screen_rect();

    // Stack from bottom up.
    let mut y = screen.max.y - TOAST_MARGIN;

    for toast in toasts.iter().rev() {
        let alpha = toast.alpha();
        if alpha == 0 { continue; }

        let a = |c: Color32| Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha);

        let accent = match toast.kind {
            ToastKind::Success  => a(p.status_ok),
            ToastKind::Error    => a(p.status_err),
            ToastKind::Info     => a(p.accent_secondary),
            ToastKind::Progress => a(p.accent_primary),
        };

        let toast_rect = Rect::from_min_size(
            egui::pos2(screen.max.x - TOAST_MARGIN - TOAST_W, y - TOAST_H),
            Vec2::new(TOAST_W, TOAST_H),
        );

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new(format!("toast_{}", toast.message)),
        ));

        // Background
        painter.rect_filled(toast_rect, egui::CornerRadius::same(5), a(p.bg_panel));
        // Left accent bar
        painter.rect_filled(
            Rect::from_min_size(toast_rect.min, Vec2::new(3.0, TOAST_H)),
            egui::CornerRadius::same(2),
            accent,
        );
        // Border
        painter.rect_stroke(
            toast_rect,
            egui::CornerRadius::same(5),
            egui::Stroke::new(1.0, a(p.border_subtle)),
            egui::StrokeKind::Inside,
        );

        // Icon
        let icon = match toast.kind {
            ToastKind::Success  => "✓",
            ToastKind::Error    => "✕",
            ToastKind::Info     => "ℹ",
            ToastKind::Progress => "⟳",
        };
        painter.text(
            egui::pos2(toast_rect.left() + 14.0, toast_rect.center().y),
            egui::Align2::CENTER_CENTER,
            icon,
            egui::FontId::proportional(13.0),
            accent,
        );

        // Message
        painter.text(
            egui::pos2(toast_rect.left() + 26.0, toast_rect.center().y),
            egui::Align2::LEFT_CENTER,
            &toast.message,
            egui::FontId::proportional(11.5),
            a(p.text_primary),
        );

        // Progress bar (for Progress kind)
        if let Some(frac) = toast.progress {
            let bar_rect = Rect::from_min_size(
                egui::pos2(toast_rect.left() + 3.0, toast_rect.max.y - 3.0),
                Vec2::new((TOAST_W - 3.0) * frac.clamp(0.0, 1.0), 2.0),
            );
            painter.rect_filled(bar_rect, egui::CornerRadius::same(1), accent);
        }

        y -= TOAST_H + TOAST_GAP;
    }

    // Request repaint while any toast is active
    if !toasts.is_empty() {
        ctx.request_repaint();
    }
}
