//! Serialize OpenXR's native queue operations with application rendering.

use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

#[derive(Clone, Default)]
pub(crate) struct GpuAccess(Arc<Mutex<()>>);

impl GpuAccess {
    pub fn lock(&self) -> MutexGuard<'_, ()> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A throttled runtime must not block the window event loop. Render another
    /// desktop frame once its queue operation has completed.
    pub fn try_lock(&self) -> Option<MutexGuard<'_, ()>> {
        match self.0.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        }
    }
}
