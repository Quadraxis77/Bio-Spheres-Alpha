//! A specimen selection outlives its GPU slot and retains its genome payload.
use crate::{genome::Genome, simulation::gpu_physics::InspectedCellData};
#[derive(Debug, Clone, Default)]
pub struct Inspection {
    pub index: Option<usize>,
    pub data: Option<InspectedCellData>,
    pub genome: Option<Genome>,
    pub dead: bool,
    pub death_pending: bool,
    pub capture_attempted: bool,
    open_pending: bool,
}
impl Inspection {
    pub fn select(&mut self, index: Option<usize>) {
        *self = Self {
            index,
            open_pending: index.is_some(),
            ..Self::default()
        };
    }
    /// Open once when a new selection has usable readings. Live refreshes must
    /// not reopen a panel the user closed; selecting the same cell again may.
    pub fn take_open_request(&mut self) -> bool {
        self.data.is_some() && std::mem::take(&mut self.open_pending)
    }
    pub fn observe(&mut self, data: InspectedCellData) {
        if self.dead {
            return;
        }
        if let Some(previous) = self.data {
            // Indices are recycled; never inspect the replacement occupant.
            if !data.is_valid() || previous.cell_id != data.cell_id {
                self.mark_dead();
                return;
            }
            if previous.genome_id != data.genome_id {
                self.genome = None;
                self.capture_attempted = false;
            }
        } else if !data.is_valid() {
            return;
        }
        self.data = Some(data);
        if data.is_dead != 0 {
            self.mark_dead();
        }
    }
    fn mark_dead(&mut self) {
        self.dead = true;
        self.death_pending = true;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn cell(id: u32) -> InspectedCellData {
        InspectedCellData {
            is_valid: 1,
            cell_id: id,
            genome_id: 7,
            nutrients: 42.0,
            ..Default::default()
        }
    }
    #[test]
    fn death_and_reused_slot_keep_selected_genome_and_last_readings() {
        let mut i = Inspection::default();
        i.select(Some(5));
        i.observe(cell(12));
        i.genome = Some(Genome::default());
        i.observe(InspectedCellData::default());
        assert!(i.dead && i.death_pending);
        assert_eq!(i.index, Some(5));
        i.death_pending = false;
        i.observe(cell(99));
        assert_eq!(i.data.unwrap().cell_id, 12);
        assert_eq!(i.data.unwrap().nutrients, 42.0);
        assert!(i.genome.is_some());
        assert!(!i.death_pending);
        i.select(Some(5));
        i.observe(cell(99));
        assert!(!i.dead);
        assert_eq!(i.data.unwrap().cell_id, 99);
        assert!(i.genome.is_none());
    }
    #[test]
    fn changed_identity_without_dead_readback_is_reported_as_death() {
        let mut i = Inspection::default();
        i.select(Some(0));
        i.observe(cell(1));
        i.observe(cell(2));
        assert!(i.dead);
        assert_eq!(i.data.unwrap().cell_id, 1);
    }
}
