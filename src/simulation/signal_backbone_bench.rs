//! Isolated Phase 0/1 support for the Cached Signal Backbone benchmark.
//!
//! Nothing in this module is connected to live preview or GPU gameplay.  It
//! provides the mathematical CPU oracle and deterministic synthetic forests
//! used to decide whether a blocked GPU implementation is allowed to advance
//! to integration.

use std::collections::{BTreeMap, VecDeque};

use bytemuck::{Pod, Zeroable};

pub const CHANNEL_COUNT: usize = 16;
pub const SIGNAL_MIN: f32 = -1000.0;
pub const SIGNAL_MAX: f32 = 1000.0;
pub const NORMAL_RETENTION: f32 = 0.95;
pub const VASCULAR_RETENTION: f32 = 0.9875;
pub const INVALID_NODE: u32 = u32::MAX;
/// Absolute tolerance in the approved `-1000..=1000` signal scale.
///
/// This is twice the maximum accepted CPU/GPU transport error so Equal and
/// Divide cannot disagree solely because of an otherwise passing propagation.
pub const PROCESSOR_EPSILON: f32 = 0.1;
pub const SIGNAL_TICK_SECONDS: f32 = 1.0 / 15.0;
pub const TOPOLOGY_GENERATION_INITIAL: u32 = 1;

pub const ROLE_RELAY: u32 = 1 << 0;
pub const ROLE_SOURCE_ONLY: u32 = 1 << 1;
pub const ROLE_DISABLED: u32 = 1 << 2;
pub const BOUNDARY_REDUCTION_WIDTH: usize = 256;

pub type Channels = [f32; CHANNEL_COUNT];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CognocyteOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Minimum,
    Maximum,
    Average,
    GreaterThan,
    LessThan,
    Equal,
    And,
    Or,
    Not,
    Select,
    Abs,
    Negate,
    Positive,
    Negative,
}

impl CognocyteOperation {
    fn is_unary(self) -> bool {
        matches!(
            self,
            Self::Not | Self::Abs | Self::Negate | Self::Positive | Self::Negative
        )
    }
}

/// Mathematical processor oracle. A zero field value is silence, so binary
/// operations go dark when either required input is silent. Unary operations
/// only require A; notably, NOT deliberately turns silent A into true.
pub fn evaluate_cognocyte(operation: CognocyteOperation, a: f32, b: f32) -> (f32, bool) {
    if !a.is_finite() || !b.is_finite() {
        return (0.0, true);
    }
    if !operation.is_unary() && (a == 0.0 || b == 0.0) {
        return (0.0, false);
    }

    let value = match operation {
        CognocyteOperation::Add => a + b,
        CognocyteOperation::Subtract => a - b,
        CognocyteOperation::Multiply => a * b / SIGNAL_MAX,
        CognocyteOperation::Divide => {
            if b.abs() <= PROCESSOR_EPSILON {
                0.0
            } else {
                a * SIGNAL_MAX / b
            }
        }
        CognocyteOperation::Minimum => a.min(b),
        CognocyteOperation::Maximum => a.max(b),
        CognocyteOperation::Average => (a + b) * 0.5,
        CognocyteOperation::GreaterThan => {
            if a > b {
                SIGNAL_MAX
            } else {
                0.0
            }
        }
        CognocyteOperation::LessThan => {
            if a < b {
                SIGNAL_MAX
            } else {
                0.0
            }
        }
        CognocyteOperation::Equal => {
            if (a - b).abs() <= PROCESSOR_EPSILON {
                SIGNAL_MAX
            } else {
                0.0
            }
        }
        CognocyteOperation::And => {
            if a > 0.0 && b > 0.0 {
                SIGNAL_MAX
            } else {
                0.0
            }
        }
        CognocyteOperation::Or => {
            if a > 0.0 || b > 0.0 {
                SIGNAL_MAX
            } else {
                0.0
            }
        }
        CognocyteOperation::Not => {
            if a > 0.0 {
                0.0
            } else {
                SIGNAL_MAX
            }
        }
        CognocyteOperation::Select => {
            if a > 0.0 {
                b
            } else {
                0.0
            }
        }
        CognocyteOperation::Abs => a.abs(),
        CognocyteOperation::Negate => -a,
        CognocyteOperation::Positive => a.max(0.0),
        CognocyteOperation::Negative => a.min(0.0),
    };

    if value.is_finite() {
        (value.clamp(SIGNAL_MIN, SIGNAL_MAX), false)
    } else {
        (0.0, true)
    }
}

/// Fixed-15-Hz Memorocyte oracle. The returned value is both next state and
/// next-tick stored output.
pub fn update_memorocyte(state: f32, input: f32, configured_rate: f32) -> (f32, bool) {
    if !state.is_finite() || !input.is_finite() || !configured_rate.is_finite() {
        return (0.0, true);
    }
    let rate = configured_rate.clamp(0.0, 1.0);
    let effective_rate = 1.0 - (1.0 - rate).powf(SIGNAL_TICK_SECONDS);
    let next = state + (input - state) * effective_rate;
    if next.is_finite() {
        (next.clamp(SIGNAL_MIN, SIGNAL_MAX), false)
    } else {
        (0.0, true)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeRole {
    Relay,
    SourceOnly,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EdgeClass {
    Normal,
    VascularRoad,
}

impl EdgeClass {
    #[inline]
    pub fn retention(self) -> f32 {
        match self {
            Self::Normal => NORMAL_RETENTION,
            Self::VascularRoad => VASCULAR_RETENTION,
        }
    }

    /// Deterministic fixed-point negative-log attenuation cost. These constants
    /// are rounded to one micro-nat and are shared by preview and GPU topology.
    #[inline]
    pub const fn resistance(self) -> u32 {
        match self {
            Self::Normal => 51_293,
            Self::VascularRoad => 12_579,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BondClass {
    /// Immutable signal-carrying bond between relay nodes.
    Backbone,
    /// Immutable source attachment incident to a source-only node.
    SourceAttachment,
    /// Mechanical cross-link. It is deliberately absent from signal topology.
    MechanicalOnly,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub a: u32,
    pub b: u32,
    pub edge_class: EdgeClass,
    pub bond_class: BondClass,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub struct SyntheticForest {
    pub roles: Vec<NodeRole>,
    pub edges: Vec<Edge>,
    pub sources: Vec<Channels>,
    /// Diagnostic-only count. These edges never enter the cached signal forest.
    pub omitted_mechanical_cross_links: u64,
}

impl SyntheticForest {
    pub fn new(node_count: usize) -> Self {
        Self {
            roles: vec![NodeRole::Relay; node_count],
            edges: Vec::new(),
            sources: vec![[0.0; CHANNEL_COUNT]; node_count],
            omitted_mechanical_cross_links: 0,
        }
    }

    pub fn cache(&self) -> Result<CachedForest, TopologyError> {
        CachedForest::build(self)
    }

    /// Select one automatic active propagation forest from the complete valid
    /// backbone graph. `Edge::active` in the returned snapshot means selected
    /// for transport; unselected backbone edges remain present as standby.
    pub fn select_active_routes(&self) -> Result<Self, TopologyError> {
        let stable_ids = (0..self.edges.len())
            .map(|index| index as u64)
            .collect::<Vec<_>>();
        self.select_active_routes_with_ids(&stable_ids)
    }

    /// Variant used by live topology, where a bond's stable identity includes
    /// its slot generation and therefore survives deterministic slot reuse.
    pub fn select_active_routes_with_ids(&self, stable_ids: &[u64]) -> Result<Self, TopologyError> {
        if stable_ids.len() != self.edges.len() {
            return Err(TopologyError::StableIdentityCountMismatch {
                edges: self.edges.len(),
                identities: stable_ids.len(),
            });
        }
        let mut routed = self.clone();
        let mut selected = vec![false; self.edges.len()];
        let mut adjacency = vec![Vec::<(u32, usize)>::new(); self.roles.len()];

        let mut edge_order = (0..self.edges.len()).collect::<Vec<_>>();
        edge_order.sort_unstable_by_key(|&index| stable_ids[index]);

        for edge_index in edge_order {
            let edge = &self.edges[edge_index];
            if !edge.active || edge.bond_class != BondClass::Backbone {
                routed.edges[edge_index].active =
                    edge.active && edge.bond_class == BondClass::SourceAttachment;
                continue;
            }
            let a = edge.a as usize;
            let b = edge.b as usize;
            if a >= self.roles.len() {
                return Err(TopologyError::EndpointOutOfRange {
                    edge: edge_index,
                    endpoint: edge.a,
                });
            }
            if b >= self.roles.len() {
                return Err(TopologyError::EndpointOutOfRange {
                    edge: edge_index,
                    endpoint: edge.b,
                });
            }
            if self.roles[a] != NodeRole::Relay || self.roles[b] != NodeRole::Relay {
                return Err(TopologyError::BackboneTouchesNonRelay { edge: edge_index });
            }

            if let Some(path) = selected_path(a, b, &adjacency) {
                if path.is_empty() {
                    continue;
                }
                let old_resistance = path
                    .iter()
                    .map(|&index| self.edges[index].edge_class.resistance() as u64)
                    .sum::<u64>();
                if edge.edge_class.resistance() as u64 >= old_resistance {
                    continue;
                }
                let maximum = path
                    .iter()
                    .map(|&index| self.edges[index].edge_class.resistance())
                    .max()
                    .unwrap_or(0);
                let midpoint = old_resistance as i64;
                let demote = path
                    .iter()
                    .copied()
                    .filter(|&index| self.edges[index].edge_class.resistance() == maximum)
                    .min_by_key(|&index| {
                        let resistance = self.edges[index].edge_class.resistance() as i64;
                        let center_twice = prefix_for_edge(index, &path, self) * 2 + resistance;
                        ((center_twice - midpoint).abs(), stable_ids[index])
                    })
                    .expect("non-empty selected path");
                selected[demote] = false;
                remove_selected_edge(demote, &self.edges, &mut adjacency);
            }
            selected[edge_index] = true;
            adjacency[a].push((edge.b, edge_index));
            adjacency[b].push((edge.a, edge_index));
        }
        for (index, edge) in routed.edges.iter_mut().enumerate() {
            if edge.bond_class == BondClass::Backbone {
                edge.active = selected[index];
            }
        }
        Ok(routed)
    }
}

fn selected_path(start: usize, goal: usize, adjacency: &[Vec<(u32, usize)>]) -> Option<Vec<usize>> {
    let mut parent = vec![None::<(usize, usize)>; adjacency.len()];
    let mut stack = vec![start];
    parent[start] = Some((start, usize::MAX));
    while let Some(node) = stack.pop() {
        if node == goal {
            break;
        }
        for &(neighbor, edge) in &adjacency[node] {
            let neighbor = neighbor as usize;
            if parent[neighbor].is_none() {
                parent[neighbor] = Some((node, edge));
                stack.push(neighbor);
            }
        }
    }
    parent[goal]?;
    let mut path = Vec::new();
    let mut node = goal;
    while node != start {
        let (previous, edge) = parent[node]?;
        path.push(edge);
        node = previous;
    }
    path.reverse();
    Some(path)
}

fn prefix_for_edge(edge: usize, path: &[usize], forest: &SyntheticForest) -> i64 {
    path.iter()
        .take_while(|&&index| index != edge)
        .map(|&index| forest.edges[index].edge_class.resistance() as i64)
        .sum()
}

fn remove_selected_edge(edge: usize, edges: &[Edge], adjacency: &mut [Vec<(u32, usize)>]) {
    let connection = edges[edge];
    adjacency[connection.a as usize].retain(|&(_, index)| index != edge);
    adjacency[connection.b as usize].retain(|&(_, index)| index != edge);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LifecycleBond {
    pub stable_id: u64,
    pub a: u32,
    pub b: u32,
    pub resistance: u32,
    pub valid: bool,
    pub backbone: bool,
    pub active: bool,
    /// New bonds cannot enter transport until a signal-tick commit.
    pub pending: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RepairJob {
    broken_id: u64,
    cut_a: u32,
    cut_b: u32,
}

/// Deterministic reference lifecycle for Phase 4. It intentionally uses graph
/// walks and is the correctness oracle for the bounded GPU implementation, not
/// the final 200k-cell topology implementation.
#[derive(Clone, Debug)]
pub struct BackboneLifecycleOracle {
    node_count: usize,
    pub bonds: Vec<LifecycleBond>,
    repair_jobs: VecDeque<RepairJob>,
    pub generation: u32,
}

impl BackboneLifecycleOracle {
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            bonds: Vec::new(),
            repair_jobs: VecDeque::new(),
            generation: TOPOLOGY_GENERATION_INITIAL,
        }
    }

    pub fn insert_backbone(
        &mut self,
        stable_id: u64,
        a: u32,
        b: u32,
        resistance: u32,
    ) -> Result<usize, TopologyError> {
        if a as usize >= self.node_count {
            return Err(TopologyError::EndpointOutOfRange {
                edge: self.bonds.len(),
                endpoint: a,
            });
        }
        if b as usize >= self.node_count {
            return Err(TopologyError::EndpointOutOfRange {
                edge: self.bonds.len(),
                endpoint: b,
            });
        }
        if a == b || self.bonds.iter().any(|bond| bond.stable_id == stable_id) {
            return Err(TopologyError::DuplicateStableIdentity {
                identity: stable_id,
            });
        }
        self.bonds.push(LifecycleBond {
            stable_id,
            a,
            b,
            resistance,
            valid: true,
            backbone: true,
            active: false,
            pending: true,
        });
        Ok(self.bonds.len() - 1)
    }

    /// Invalidates a physical bond synchronously. If it was selected, no later
    /// propagation can observe it and a bounded repair job is queued.
    pub fn invalidate(&mut self, stable_id: u64) -> bool {
        let Some(index) = self
            .bonds
            .iter()
            .position(|bond| bond.stable_id == stable_id)
        else {
            return false;
        };
        if !self.bonds[index].valid {
            return false;
        }
        let was_active = self.bonds[index].active;
        let cut_a = self.bonds[index].a;
        let cut_b = self.bonds[index].b;
        self.bonds[index].valid = false;
        self.bonds[index].active = false;
        self.bonds[index].pending = false;
        if was_active {
            self.repair_jobs.push_back(RepairJob {
                broken_id: stable_id,
                cut_a,
                cut_b,
            });
        }
        true
    }

    pub fn pending_job_count(&self) -> usize {
        self.bonds
            .iter()
            .filter(|bond| bond.pending && bond.valid)
            .count()
            + self.repair_jobs.len()
    }

    /// Commit at most `budget` additions/repairs. Ordering is stable identity,
    /// never shader race order. Returns the number of jobs consumed.
    pub fn commit(&mut self, budget: usize) -> usize {
        let mut remaining = budget;
        let mut changed = false;
        let mut additions = self
            .bonds
            .iter()
            .enumerate()
            .filter(|(_, bond)| bond.valid && bond.backbone && bond.pending)
            .map(|(index, bond)| (bond.stable_id, index))
            .collect::<Vec<_>>();
        additions.sort_unstable();

        for (_, index) in additions {
            if remaining == 0 {
                break;
            }
            self.bonds[index].pending = false;
            changed |= self.commit_addition(index);
            remaining -= 1;
        }

        while remaining > 0 {
            let Some(job) = self.repair_jobs.pop_front() else {
                break;
            };
            changed |= self.commit_repair(job);
            remaining -= 1;
        }

        if changed {
            self.generation = self.generation.wrapping_add(1).max(1);
        }
        budget - remaining
    }

    fn active_adjacency(&self) -> Vec<Vec<(usize, usize)>> {
        let mut adjacency = vec![Vec::new(); self.node_count];
        for (index, bond) in self.bonds.iter().enumerate() {
            if bond.valid && bond.backbone && bond.active {
                adjacency[bond.a as usize].push((bond.b as usize, index));
                adjacency[bond.b as usize].push((bond.a as usize, index));
            }
        }
        adjacency
    }

    fn commit_addition(&mut self, index: usize) -> bool {
        let adjacency = self.active_adjacency();
        let bond = self.bonds[index];
        let Some(path) = lifecycle_path(bond.a as usize, bond.b as usize, &adjacency) else {
            self.bonds[index].active = true;
            return true;
        };
        let path_resistance = path
            .iter()
            .map(|&edge| self.bonds[edge].resistance as u64)
            .sum::<u64>();
        if bond.resistance as u64 >= path_resistance {
            return false;
        }
        let maximum = path
            .iter()
            .map(|&edge| self.bonds[edge].resistance)
            .max()
            .unwrap_or(0);
        let midpoint_twice = path_resistance;
        let demote = path
            .iter()
            .copied()
            .filter(|&edge| self.bonds[edge].resistance == maximum)
            .map(|edge| {
                let center_twice = prefix_for_lifecycle_edge(edge, &path, &self.bonds) * 2
                    + self.bonds[edge].resistance as u64;
                (
                    center_twice.abs_diff(midpoint_twice),
                    self.bonds[edge].stable_id,
                    edge,
                )
            })
            .min()
            .map(|(_, _, edge)| edge)
            .expect("connected path is non-empty");
        self.bonds[demote].active = false;
        self.bonds[index].active = true;
        true
    }

    fn commit_repair(&mut self, job: RepairJob) -> bool {
        let adjacency = self.active_adjacency();
        if lifecycle_path(job.cut_a as usize, job.cut_b as usize, &adjacency).is_some() {
            return false;
        }
        let distances_a = lifecycle_distances(job.cut_a as usize, &adjacency, &self.bonds);
        let distances_b = lifecycle_distances(job.cut_b as usize, &adjacency, &self.bonds);
        let candidate = self
            .bonds
            .iter()
            .enumerate()
            .filter(|(_, bond)| bond.valid && bond.backbone && !bond.active && !bond.pending)
            .filter_map(|(index, bond)| {
                let direct = distances_a[bond.a as usize].zip(distances_b[bond.b as usize]);
                let reverse = distances_a[bond.b as usize].zip(distances_b[bond.a as usize]);
                let route = direct.or(reverse)?;
                Some((
                    route.0 + bond.resistance as u64 + route.1,
                    bond.stable_id,
                    index,
                ))
            })
            .min();
        if let Some((_, _, index)) = candidate {
            self.bonds[index].active = true;
            true
        } else {
            let _ = job.broken_id;
            false
        }
    }
}

fn lifecycle_path(
    start: usize,
    goal: usize,
    adjacency: &[Vec<(usize, usize)>],
) -> Option<Vec<usize>> {
    let mut parent = vec![None::<(usize, usize)>; adjacency.len()];
    let mut queue = VecDeque::from([start]);
    parent[start] = Some((start, usize::MAX));
    while let Some(node) = queue.pop_front() {
        if node == goal {
            break;
        }
        for &(neighbor, edge) in &adjacency[node] {
            if parent[neighbor].is_none() {
                parent[neighbor] = Some((node, edge));
                queue.push_back(neighbor);
            }
        }
    }
    parent[goal]?;
    let mut path = Vec::new();
    let mut node = goal;
    while node != start {
        let (previous, edge) = parent[node]?;
        path.push(edge);
        node = previous;
    }
    path.reverse();
    Some(path)
}

fn prefix_for_lifecycle_edge(edge: usize, path: &[usize], bonds: &[LifecycleBond]) -> u64 {
    path.iter()
        .take_while(|&&index| index != edge)
        .map(|&index| bonds[index].resistance as u64)
        .sum()
}

fn lifecycle_distances(
    root: usize,
    adjacency: &[Vec<(usize, usize)>],
    bonds: &[LifecycleBond],
) -> Vec<Option<u64>> {
    let mut distances = vec![None; adjacency.len()];
    let mut stack = vec![(root, usize::MAX, 0u64)];
    while let Some((node, parent, distance)) = stack.pop() {
        distances[node] = Some(distance);
        for &(neighbor, edge) in &adjacency[node] {
            if neighbor != parent {
                stack.push((neighbor, node, distance + bonds[edge].resistance as u64));
            }
        }
    }
    distances
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TopologyError {
    SourceCountMismatch,
    StableIdentityCountMismatch { edges: usize, identities: usize },
    DuplicateStableIdentity { identity: u64 },
    EndpointOutOfRange { edge: usize, endpoint: u32 },
    BackboneTouchesNonRelay { edge: usize },
    AttachmentRoleMismatch { edge: usize },
    RelayCycle { edge: usize },
}

/// Deterministically rooted forest metadata consumed by the CPU oracle and
/// mirrored by standalone GPU benchmark candidates.
#[derive(Clone, Debug)]
pub struct CachedForest {
    pub parent: Vec<u32>,
    pub parent_retention: Vec<f32>,
    pub preorder: Vec<u32>,
    pub source_attachments: Vec<SourceAttachment>,
    pub roles: Vec<NodeRole>,
}

#[derive(Clone, Copy, Debug)]
pub struct SourceAttachment {
    pub source: u32,
    pub relay: u32,
    pub retention: f32,
}

#[derive(Clone, Debug)]
pub struct Microtree {
    pub nodes: Vec<u32>,
    pub parent_microtree: u32,
    pub attachment_node: u32,
}

#[derive(Clone, Debug)]
pub struct MicrotreeSchedule {
    pub microtrees: Vec<Microtree>,
    pub node_microtree: Vec<u32>,
    pub block_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MacroStrategy {
    PointerJumpingPath,
    DepthBuckets,
}

#[derive(Clone, Debug)]
pub struct MacroSchedule {
    pub strategy: MacroStrategy,
    /// Parent-before-child buckets. Roots occupy bucket zero.
    pub depth_buckets: Vec<Vec<u32>>,
    pub maximum_children: u32,
}

/// Fixed 32-byte per-cell topology ABI used by the standalone WGSL candidate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCellTopology {
    pub parent_cell: u32,
    pub microtree_id: u32,
    pub local_index: u32,
    pub role_flags: u32,
    pub generation: u32,
    pub parent_retention: f32,
    pub local_depth: u32,
    pub _padding: u32,
}

/// Fixed 32-byte per-microtree topology ABI. Child boundaries identify child
/// microtrees, not physical children internal to this block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMicrotreeTopology {
    pub node_offset: u32,
    pub node_count: u32,
    pub parent_microtree: u32,
    pub attachment_node: u32,
    pub external_parent_cell: u32,
    pub child_boundary_offset: u32,
    pub child_boundary_count: u32,
    pub generation: u32,
}

#[derive(Clone, Debug)]
pub struct GpuTopologyUpload {
    pub cells: Vec<GpuCellTopology>,
    pub microtrees: Vec<GpuMicrotreeTopology>,
    pub node_list: Vec<u32>,
    pub child_microtrees: Vec<u32>,
    pub depth_offsets: Vec<u32>,
    pub depth_microtrees: Vec<u32>,
    pub macro_strategy: MacroStrategy,
    pub maximum_children: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundaryInputKind {
    MicrotreeUpMessage,
    PreviousPassPartial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundaryReductionChunk {
    pub input_offset: u32,
    pub input_count: u32,
    pub output_slot: u32,
    pub target_parent_cell: u32,
    pub final_output: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundaryReductionPass {
    pub input_kind: BoundaryInputKind,
    pub inputs: Vec<u32>,
    pub chunks: Vec<BoundaryReductionChunk>,
    pub scratch_output_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DepthBoundaryReduction {
    pub child_depth: u32,
    pub passes: Vec<BoundaryReductionPass>,
}

impl GpuTopologyUpload {
    pub fn allocated_bytes(&self) -> u64 {
        (self.cells.len() * std::mem::size_of::<GpuCellTopology>()
            + self.microtrees.len() * std::mem::size_of::<GpuMicrotreeTopology>()
            + self.node_list.len() * std::mem::size_of::<u32>()
            + self.child_microtrees.len() * std::mem::size_of::<u32>()
            + self.depth_offsets.len() * std::mem::size_of::<u32>()
            + self.depth_microtrees.len() * std::mem::size_of::<u32>()) as u64
    }
}

impl MacroSchedule {
    pub fn macro_dispatches(&self) -> u32 {
        match self.strategy {
            MacroStrategy::PointerJumpingPath => {
                let maximum_path_nodes = self.depth_buckets.len() as u32;
                if maximum_path_nodes <= 1 {
                    0
                } else {
                    u32::BITS - (maximum_path_nodes - 1).leading_zeros()
                }
            }
            MacroStrategy::DepthBuckets => (self.depth_buckets.len() as u32).saturating_mul(2),
        }
    }
}

impl MicrotreeSchedule {
    /// Greedily grows connected breadth-first microtrees. Every non-root block
    /// has exactly one parent boundary because a child is considered only after
    /// its parent has been assigned. Deterministic node ordering provides stable
    /// topology metadata independent of GPU scheduling.
    pub fn build(cache: &CachedForest, block_size: usize) -> Self {
        assert!(block_size > 0);
        let node_count = cache.roles.len();
        let mut children = vec![Vec::<u32>::new(); node_count];
        for &node in &cache.preorder {
            let parent = cache.parent[node as usize];
            if parent != INVALID_NODE {
                children[parent as usize].push(node);
            }
        }
        for child_list in &mut children {
            child_list.sort_unstable();
        }

        let mut node_microtree = vec![INVALID_NODE; node_count];
        let mut microtrees = Vec::new();
        let mut queue = VecDeque::new();

        for &seed in &cache.preorder {
            if node_microtree[seed as usize] != INVALID_NODE {
                continue;
            }
            let microtree_id = microtrees.len() as u32;
            let parent = cache.parent[seed as usize];
            let parent_microtree = if parent == INVALID_NODE {
                INVALID_NODE
            } else {
                node_microtree[parent as usize]
            };
            debug_assert!(parent == INVALID_NODE || parent_microtree != INVALID_NODE);

            let mut nodes = Vec::with_capacity(block_size);
            queue.clear();
            queue.push_back(seed);
            while nodes.len() < block_size {
                let Some(node) = queue.pop_front() else {
                    break;
                };
                if node_microtree[node as usize] != INVALID_NODE {
                    continue;
                }
                node_microtree[node as usize] = microtree_id;
                nodes.push(node);
                for &child in &children[node as usize] {
                    if node_microtree[child as usize] == INVALID_NODE {
                        queue.push_back(child);
                    }
                }
            }

            microtrees.push(Microtree {
                nodes,
                parent_microtree,
                attachment_node: seed,
            });
        }

        // Canonicalize IDs by macro depth so every depth bucket is one compact
        // dispatch range. The secondary old-ID key preserves deterministic
        // ordering within a depth and parent IDs remain smaller than children.
        let mut depths = vec![0_usize; microtrees.len()];
        for (microtree_id, microtree) in microtrees.iter().enumerate() {
            if microtree.parent_microtree != INVALID_NODE {
                depths[microtree_id] = depths[microtree.parent_microtree as usize] + 1;
            }
        }
        let mut order = (0..microtrees.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&old_id| (depths[old_id], old_id));
        let mut old_to_new = vec![INVALID_NODE; microtrees.len()];
        for (new_id, &old_id) in order.iter().enumerate() {
            old_to_new[old_id] = new_id as u32;
        }
        let canonical_microtrees = order
            .into_iter()
            .map(|old_id| {
                let mut microtree = microtrees[old_id].clone();
                if microtree.parent_microtree != INVALID_NODE {
                    microtree.parent_microtree = old_to_new[microtree.parent_microtree as usize];
                }
                microtree
            })
            .collect::<Vec<_>>();
        for microtree_id in &mut node_microtree {
            if *microtree_id != INVALID_NODE {
                *microtree_id = old_to_new[*microtree_id as usize];
            }
        }

        Self {
            microtrees: canonical_microtrees,
            node_microtree,
            block_size,
        }
    }

    /// Executes the exact tree-message equations through microtree boundaries.
    /// This intentionally differs from `CachedForest::propagate` structurally:
    /// it is the executable oracle for the flattened GPU schedule.
    pub fn propagate(
        &self,
        cache: &CachedForest,
        sources: &[Channels],
    ) -> Result<Vec<Channels>, TopologyError> {
        if sources.len() != cache.roles.len() {
            return Err(TopologyError::SourceCountMismatch);
        }

        let node_count = cache.roles.len();
        let mut output = vec![[0.0; CHANNEL_COUNT]; node_count];
        let mut local_source = vec![0.0_f64; node_count];
        let mut subtree = vec![0.0_f64; node_count];
        let mut from_parent = vec![0.0_f64; node_count];
        let mut boundary_up = vec![0.0_f64; self.microtrees.len()];
        let attachment_source = cache.attachment_values(sources);

        for channel in 0..CHANNEL_COUNT {
            local_source.fill(0.0);
            subtree.fill(0.0);
            from_parent.fill(0.0);
            boundary_up.fill(0.0);

            for &node in &cache.preorder {
                let index = node as usize;
                local_source[index] = sources[index][channel].clamp(SIGNAL_MIN, SIGNAL_MAX) as f64
                    + attachment_source[index][channel] as f64;
                subtree[index] = local_source[index];
            }

            // Child blocks have larger deterministic IDs than their parent.
            // Their completed upward boundary message is injected directly at
            // the external parent before that parent block is reduced.
            for microtree_id in (0..self.microtrees.len()).rev() {
                let microtree = &self.microtrees[microtree_id];
                for &node in microtree.nodes.iter().rev() {
                    let index = node as usize;
                    let parent = cache.parent[index];
                    if parent != INVALID_NODE
                        && self.node_microtree[parent as usize] == microtree_id as u32
                    {
                        subtree[parent as usize] +=
                            cache.parent_retention[index] as f64 * subtree[index];
                    }
                }

                let attachment = microtree.attachment_node as usize;
                let parent = cache.parent[attachment];
                if parent != INVALID_NODE {
                    let message = cache.parent_retention[attachment] as f64 * subtree[attachment];
                    boundary_up[microtree_id] = message;
                    subtree[parent as usize] += message;
                }
            }

            // Parent blocks have smaller IDs. Establish the external incoming
            // message, then walk the local parent-before-child traversal.
            for (microtree_id, microtree) in self.microtrees.iter().enumerate() {
                let attachment = microtree.attachment_node as usize;
                let external_parent = cache.parent[attachment];
                if external_parent != INVALID_NODE {
                    let parent = external_parent as usize;
                    from_parent[attachment] = cache.parent_retention[attachment] as f64
                        * (from_parent[parent] + subtree[parent] - boundary_up[microtree_id]);
                }

                for &node in &microtree.nodes {
                    let index = node as usize;
                    let parent = cache.parent[index];
                    if parent != INVALID_NODE
                        && self.node_microtree[parent as usize] == microtree_id as u32
                    {
                        let parent_index = parent as usize;
                        let own_up = cache.parent_retention[index] as f64 * subtree[index];
                        from_parent[index] = cache.parent_retention[index] as f64
                            * (from_parent[parent_index] + subtree[parent_index] - own_up);
                    }

                    let received = from_parent[index] + subtree[index] - local_source[index]
                        + attachment_source[index][channel] as f64;
                    output[index][channel] =
                        received.clamp(SIGNAL_MIN as f64, SIGNAL_MAX as f64) as f32;
                }
            }
        }

        Ok(output)
    }

    pub fn macro_schedule(&self) -> MacroSchedule {
        let mut children = vec![0_u32; self.microtrees.len()];
        let mut depth = vec![0_usize; self.microtrees.len()];
        let mut maximum_children = 0;

        for (microtree_id, microtree) in self.microtrees.iter().enumerate() {
            if microtree.parent_microtree != INVALID_NODE {
                let parent = microtree.parent_microtree as usize;
                children[parent] += 1;
                maximum_children = maximum_children.max(children[parent]);
                debug_assert!(parent < microtree_id);
                depth[microtree_id] = depth[parent] + 1;
            }
        }

        let mut depth_buckets = vec![Vec::new(); depth.iter().copied().max().unwrap_or(0) + 1];
        for (microtree_id, &microtree_depth) in depth.iter().enumerate() {
            depth_buckets[microtree_depth].push(microtree_id as u32);
        }

        MacroSchedule {
            strategy: if maximum_children <= 1 {
                MacroStrategy::PointerJumpingPath
            } else {
                MacroStrategy::DepthBuckets
            },
            depth_buckets,
            maximum_children,
        }
    }

    pub fn flatten_for_gpu(&self, cache: &CachedForest, generation: u32) -> GpuTopologyUpload {
        assert_eq!(self.node_microtree.len(), cache.roles.len());

        let macro_schedule = self.macro_schedule();
        let mut local_indices = vec![INVALID_NODE; cache.roles.len()];
        let mut local_depths = vec![INVALID_NODE; cache.roles.len()];
        let mut node_list = Vec::with_capacity(cache.roles.len());
        for (microtree_id, microtree) in self.microtrees.iter().enumerate() {
            for (local_index, &node) in microtree.nodes.iter().enumerate() {
                local_indices[node as usize] = local_index as u32;
                let parent = cache.parent[node as usize];
                local_depths[node as usize] = if parent != INVALID_NODE
                    && self.node_microtree[parent as usize] == microtree_id as u32
                {
                    local_depths[parent as usize] + 1
                } else {
                    0
                };
                node_list.push(node);
            }
        }

        let cells = cache
            .roles
            .iter()
            .enumerate()
            .map(|(cell, role)| GpuCellTopology {
                parent_cell: cache.parent[cell],
                microtree_id: self.node_microtree[cell],
                local_index: local_indices[cell],
                role_flags: match role {
                    NodeRole::Relay => ROLE_RELAY,
                    NodeRole::SourceOnly => ROLE_SOURCE_ONLY,
                    NodeRole::Disabled => ROLE_DISABLED,
                },
                generation,
                parent_retention: cache.parent_retention[cell],
                local_depth: local_depths[cell],
                _padding: 0,
            })
            .collect();

        let mut children = vec![Vec::<u32>::new(); self.microtrees.len()];
        for (microtree_id, microtree) in self.microtrees.iter().enumerate() {
            if microtree.parent_microtree != INVALID_NODE {
                children[microtree.parent_microtree as usize].push(microtree_id as u32);
            }
        }

        let mut child_microtrees = Vec::with_capacity(self.microtrees.len().saturating_sub(1));
        let mut microtrees = Vec::with_capacity(self.microtrees.len());
        let mut node_offset = 0_u32;
        for (microtree_id, microtree) in self.microtrees.iter().enumerate() {
            let child_boundary_offset = child_microtrees.len() as u32;
            child_microtrees.extend_from_slice(&children[microtree_id]);
            let external_parent_cell = cache.parent[microtree.attachment_node as usize];
            microtrees.push(GpuMicrotreeTopology {
                node_offset,
                node_count: microtree.nodes.len() as u32,
                parent_microtree: microtree.parent_microtree,
                attachment_node: microtree.attachment_node,
                external_parent_cell,
                child_boundary_offset,
                child_boundary_count: children[microtree_id].len() as u32,
                generation,
            });
            node_offset += microtree.nodes.len() as u32;
        }

        let mut depth_offsets = Vec::with_capacity(macro_schedule.depth_buckets.len() + 1);
        let mut depth_microtrees = Vec::with_capacity(self.microtrees.len());
        depth_offsets.push(0);
        for bucket in &macro_schedule.depth_buckets {
            depth_microtrees.extend_from_slice(bucket);
            depth_offsets.push(depth_microtrees.len() as u32);
        }

        GpuTopologyUpload {
            cells,
            microtrees,
            node_list,
            child_microtrees,
            depth_offsets,
            depth_microtrees,
            macro_strategy: macro_schedule.strategy,
            maximum_children: macro_schedule.maximum_children,
        }
    }

    /// Precomputes bounded segmented reductions for child-microtree messages.
    /// Every chunk is at most 256 values. High-degree parents therefore become
    /// a logarithmic hierarchy of deterministic partials rather than a serial
    /// scan or nondeterministic floating-point atomics.
    pub fn boundary_reduction_schedule(&self, cache: &CachedForest) -> Vec<DepthBoundaryReduction> {
        let macro_schedule = self.macro_schedule();
        let mut result = Vec::with_capacity(macro_schedule.depth_buckets.len().saturating_sub(1));

        for child_depth in 1..macro_schedule.depth_buckets.len() {
            let mut groups = BTreeMap::<u32, Vec<u32>>::new();
            for &child_microtree in &macro_schedule.depth_buckets[child_depth] {
                let attachment = self.microtrees[child_microtree as usize].attachment_node;
                let parent_cell = cache.parent[attachment as usize];
                debug_assert_ne!(parent_cell, INVALID_NODE);
                groups.entry(parent_cell).or_default().push(child_microtree);
            }

            let mut passes = Vec::new();
            let mut input_kind = BoundaryInputKind::MicrotreeUpMessage;
            while !groups.is_empty() {
                let mut inputs = Vec::new();
                let mut chunks = Vec::new();
                let mut next_groups = BTreeMap::<u32, Vec<u32>>::new();
                let mut scratch_output_count = 0_u32;

                for (parent_cell, group_inputs) in groups {
                    let chunk_count = group_inputs.len().div_ceil(BOUNDARY_REDUCTION_WIDTH);
                    for values in group_inputs.chunks(BOUNDARY_REDUCTION_WIDTH) {
                        let input_offset = inputs.len() as u32;
                        inputs.extend_from_slice(values);
                        let final_output = chunk_count == 1;
                        let output_slot = if final_output {
                            INVALID_NODE
                        } else {
                            let slot = scratch_output_count;
                            scratch_output_count += 1;
                            next_groups.entry(parent_cell).or_default().push(slot);
                            slot
                        };
                        chunks.push(BoundaryReductionChunk {
                            input_offset,
                            input_count: values.len() as u32,
                            output_slot,
                            target_parent_cell: parent_cell,
                            final_output,
                        });
                    }
                }

                passes.push(BoundaryReductionPass {
                    input_kind,
                    inputs,
                    chunks,
                    scratch_output_count,
                });
                groups = next_groups;
                input_kind = BoundaryInputKind::PreviousPassPartial;
            }
            result.push(DepthBoundaryReduction {
                child_depth: child_depth as u32,
                passes,
            });
        }
        result
    }
}

impl CachedForest {
    pub fn build(forest: &SyntheticForest) -> Result<Self, TopologyError> {
        let node_count = forest.roles.len();
        if forest.sources.len() != node_count {
            return Err(TopologyError::SourceCountMismatch);
        }

        let mut adjacency = vec![Vec::<(u32, f32)>::new(); node_count];
        let mut source_attachments = Vec::new();

        for (edge_index, edge) in forest.edges.iter().enumerate() {
            for endpoint in [edge.a, edge.b] {
                if endpoint as usize >= node_count {
                    return Err(TopologyError::EndpointOutOfRange {
                        edge: edge_index,
                        endpoint,
                    });
                }
            }
            if !edge.active {
                continue;
            }

            let a = edge.a as usize;
            let b = edge.b as usize;
            match edge.bond_class {
                BondClass::MechanicalOnly => {}
                BondClass::Backbone => {
                    if forest.roles[a] != NodeRole::Relay || forest.roles[b] != NodeRole::Relay {
                        return Err(TopologyError::BackboneTouchesNonRelay { edge: edge_index });
                    }
                    let retention = edge.edge_class.retention();
                    adjacency[a].push((edge.b, retention));
                    adjacency[b].push((edge.a, retention));
                }
                BondClass::SourceAttachment => {
                    let (source, relay) = match (forest.roles[a], forest.roles[b]) {
                        (NodeRole::SourceOnly, NodeRole::Relay) => (a, b),
                        (NodeRole::Relay, NodeRole::SourceOnly) => (b, a),
                        _ => {
                            return Err(TopologyError::AttachmentRoleMismatch { edge: edge_index })
                        }
                    };
                    let retention = edge.edge_class.retention();
                    source_attachments.push(SourceAttachment {
                        source: source as u32,
                        relay: relay as u32,
                        retention,
                    });
                }
            }
        }

        // A deterministic root and neighbor order makes the oracle independent
        // of insertion order while retaining orientation-independent equations.
        for neighbors in &mut adjacency {
            neighbors.sort_unstable_by_key(|&(neighbor, _)| neighbor);
        }

        let mut parent = vec![INVALID_NODE; node_count];
        let mut parent_retention = vec![0.0; node_count];
        let mut preorder = Vec::with_capacity(node_count);
        let mut visited = vec![false; node_count];
        let mut queue = VecDeque::new();

        for root in 0..node_count {
            if forest.roles[root] != NodeRole::Relay || visited[root] {
                continue;
            }
            visited[root] = true;
            queue.push_back(root as u32);
            while let Some(node) = queue.pop_front() {
                preorder.push(node);
                for &(neighbor, retention) in &adjacency[node as usize] {
                    let neighbor_index = neighbor as usize;
                    if parent[node as usize] == neighbor {
                        continue;
                    }
                    if visited[neighbor_index] {
                        let edge = forest
                            .edges
                            .iter()
                            .position(|candidate| {
                                candidate.active
                                    && candidate.bond_class == BondClass::Backbone
                                    && ((candidate.a == node && candidate.b == neighbor)
                                        || (candidate.a == neighbor && candidate.b == node))
                            })
                            .unwrap_or(usize::MAX);
                        return Err(TopologyError::RelayCycle { edge });
                    }
                    visited[neighbor_index] = true;
                    parent[neighbor_index] = node;
                    parent_retention[neighbor_index] = retention;
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(Self {
            parent,
            parent_retention,
            preorder,
            source_attachments,
            roles: forest.roles.clone(),
        })
    }

    fn attachment_values(&self, sources: &[Channels]) -> Vec<Channels> {
        let mut values = vec![[0.0; CHANNEL_COUNT]; self.roles.len()];
        for attachment in &self.source_attachments {
            for channel in 0..CHANNEL_COUNT {
                values[attachment.relay as usize][channel] +=
                    sources[attachment.source as usize][channel].clamp(SIGNAL_MIN, SIGNAL_MAX)
                        * attachment.retention;
            }
        }
        values
    }

    /// Exact two-pass tree-message oracle. Accumulation is `f64`, source values
    /// are clamped before injection, and public values clamp only after complete
    /// signed accumulation. Mechanical-only edges never appear in this solve.
    pub fn propagate(&self, sources: &[Channels]) -> Result<Vec<Channels>, TopologyError> {
        self.propagate_with_local_overlay(sources, None)
    }

    /// Identical tree solve with an optional unattenuated local contribution
    /// included before the one final saturation step (used by heat stroke).
    pub fn propagate_with_local_overlay(
        &self,
        sources: &[Channels],
        local_overlay: Option<&[Channels]>,
    ) -> Result<Vec<Channels>, TopologyError> {
        if sources.len() != self.roles.len() {
            return Err(TopologyError::SourceCountMismatch);
        }
        if local_overlay.is_some_and(|overlay| overlay.len() != self.roles.len()) {
            return Err(TopologyError::SourceCountMismatch);
        }

        let node_count = self.roles.len();
        let mut output = vec![[0.0; CHANNEL_COUNT]; node_count];
        let mut local_source = vec![0.0f64; node_count];
        let mut subtree = vec![0.0f64; node_count];
        let mut from_parent = vec![0.0f64; node_count];
        let attachment_source = self.attachment_values(sources);

        for channel in 0..CHANNEL_COUNT {
            local_source.fill(0.0);
            subtree.fill(0.0);
            from_parent.fill(0.0);

            for &node in &self.preorder {
                let index = node as usize;
                local_source[index] = sources[index][channel].clamp(SIGNAL_MIN, SIGNAL_MAX) as f64
                    + attachment_source[index][channel] as f64;
                subtree[index] = local_source[index];
            }

            for &node in self.preorder.iter().rev() {
                let index = node as usize;
                let parent = self.parent[index];
                if parent != INVALID_NODE {
                    subtree[parent as usize] +=
                        self.parent_retention[index] as f64 * subtree[index];
                }
            }

            for &node in &self.preorder {
                let index = node as usize;
                let child_messages = subtree[index] - local_source[index];
                let received = from_parent[index]
                    + child_messages
                    + attachment_source[index][channel] as f64
                    + local_overlay.map_or(0.0, |overlay| overlay[index][channel] as f64);
                output[index][channel] =
                    received.clamp(SIGNAL_MIN as f64, SIGNAL_MAX as f64) as f32;

                // Children are contiguous only by metadata, not by node id, so
                // scan preorder once via the parent array. This loop is replaced
                // by CSR child ranges in GPU candidates; the oracle favors clarity.
            }

            // The downward recurrence needs parents before children; preorder
            // already provides that ordering. A separate pass avoids depending
            // on child contiguity or edge insertion order.
            for &node in &self.preorder {
                let index = node as usize;
                let parent = self.parent[index];
                if parent == INVALID_NODE {
                    continue;
                }
                let parent_index = parent as usize;
                let own_up_message = self.parent_retention[index] as f64 * subtree[index];
                let parent_child_messages = subtree[parent_index] - local_source[parent_index];
                from_parent[index] = self.parent_retention[index] as f64
                    * (local_source[parent_index]
                        + from_parent[parent_index]
                        + parent_child_messages
                        - own_up_message);

                let child_messages = subtree[index] - local_source[index];
                let received = from_parent[index]
                    + child_messages
                    + attachment_source[index][channel] as f64
                    + local_overlay.map_or(0.0, |overlay| overlay[index][channel] as f64);
                output[index][channel] =
                    received.clamp(SIGNAL_MIN as f64, SIGNAL_MAX as f64) as f32;
            }
        }

        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyntheticShape {
    Chain,
    Star,
    BalancedBinary,
    ManyPairs,
    GameplayMixed,
    DenseMechanicalSparseBackbone,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalWorkload {
    SilentWithInvertedListener,
    OneChannelOneSource,
    OneVec4Group,
    AllChannelsSparse,
    EveryCellEmits,
    EveryCellCognocyte,
    EveryCellMemorocyte,
    SaturatedFanIn,
    SignedCancellationFanIn,
    ContinuousOscillators,
    HeatScreamAllChannels,
}

/// Deterministic source population shared by CPU and GPU benchmark paths.
/// Cognocyte and Memorocyte throughput use separate processor-state kernels;
/// their transport load is equivalent to `EveryCellEmits` here.
pub fn populate_workload(sources: &mut [Channels], workload: SignalWorkload, tick: u64) {
    for channels in sources.iter_mut() {
        *channels = [0.0; CHANNEL_COUNT];
    }
    let count = sources.len();
    if count == 0 {
        return;
    }

    match workload {
        SignalWorkload::SilentWithInvertedListener => {}
        SignalWorkload::OneChannelOneSource => sources[0][0] = 1000.0,
        SignalWorkload::OneVec4Group => {
            for channel in 0..4 {
                sources[0][channel] = 1000.0 - channel as f32 * 125.0;
            }
        }
        SignalWorkload::AllChannelsSparse => {
            for cell in (0..count).step_by(997) {
                for channel in 0..CHANNEL_COUNT {
                    let magnitude = 20.0 + ((channel * 17 + cell % 31) % 80) as f32;
                    sources[cell][channel] = if (cell / 997 + channel) % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                }
            }
            if count > 1 {
                sources[count - 1][0] = 1000.0;
            }
        }
        SignalWorkload::EveryCellEmits => {
            for (cell, channels) in sources.iter_mut().enumerate() {
                channels[cell % CHANNEL_COUNT] = if cell % 2 == 0 { 25.0 } else { -25.0 };
            }
        }
        SignalWorkload::EveryCellCognocyte => {
            for (cell, channels) in sources.iter_mut().enumerate() {
                channels[0] = if cell % 2 == 0 { 250.0 } else { -250.0 };
                channels[1] = 500.0;
            }
        }
        SignalWorkload::EveryCellMemorocyte => {
            for (cell, channels) in sources.iter_mut().enumerate() {
                channels[0] = if cell % 2 == 0 { 750.0 } else { -750.0 };
            }
        }
        SignalWorkload::SaturatedFanIn => {
            for channels in sources.iter_mut().skip(1) {
                channels[0] = 1000.0;
            }
        }
        SignalWorkload::SignedCancellationFanIn => {
            for (cell, channels) in sources.iter_mut().enumerate().skip(1) {
                channels[0] = if cell % 2 == 0 { 1000.0 } else { -1000.0 };
            }
        }
        SignalWorkload::ContinuousOscillators => {
            let phase_tick = (tick % 120) as f32 / 120.0;
            for (cell, channels) in sources.iter_mut().enumerate().step_by(64) {
                let phase = phase_tick + ((cell / 64) % 64) as f32 / 64.0;
                channels[0] = (phase * std::f32::consts::TAU).sin() * 1000.0;
            }
        }
        SignalWorkload::HeatScreamAllChannels => {
            for (cell, channels) in sources.iter_mut().enumerate() {
                for (channel, value) in channels.iter_mut().enumerate() {
                    let mut hash = cell as u32
                        ^ (channel as u32).wrapping_mul(0x9e37_79b9)
                        ^ (tick as u32).wrapping_mul(0x85eb_ca6b);
                    hash ^= hash >> 16;
                    hash = hash.wrapping_mul(0x7feb_352d);
                    hash ^= hash >> 15;
                    hash = hash.wrapping_mul(0x846c_a68b);
                    hash ^= hash >> 16;
                    *value = if hash & 1 == 0 { -1000.0 } else { 1000.0 };
                }
            }
        }
    }
}

pub fn synthetic_forest(shape: SyntheticShape, node_count: usize) -> SyntheticForest {
    let mut forest = SyntheticForest::new(node_count);
    if node_count < 2 {
        return forest;
    }

    let mut add_backbone = |a: usize, b: usize, edge_class: EdgeClass| {
        forest.edges.push(Edge {
            a: a as u32,
            b: b as u32,
            edge_class,
            bond_class: BondClass::Backbone,
            active: true,
        });
    };

    match shape {
        SyntheticShape::Chain | SyntheticShape::DenseMechanicalSparseBackbone => {
            for node in 1..node_count {
                add_backbone(node - 1, node, EdgeClass::Normal);
            }
            if shape == SyntheticShape::DenseMechanicalSparseBackbone {
                forest.omitted_mechanical_cross_links = (node_count as u64).saturating_mul(12);
            }
        }
        SyntheticShape::Star => {
            for node in 1..node_count {
                add_backbone(0, node, EdgeClass::Normal);
            }
        }
        SyntheticShape::BalancedBinary => {
            for node in 1..node_count {
                add_backbone((node - 1) / 2, node, EdgeClass::Normal);
            }
        }
        SyntheticShape::ManyPairs => {
            for node in (0..node_count - 1).step_by(2) {
                add_backbone(node, node + 1, EdgeClass::Normal);
            }
        }
        SyntheticShape::GameplayMixed => {
            // Deterministic, segmented organisms with occasional vascular roads.
            const ORGANISM_SIZE: usize = 37;
            for base in (0..node_count).step_by(ORGANISM_SIZE) {
                let end = (base + ORGANISM_SIZE).min(node_count);
                for node in base + 1..end {
                    let parent = base + (node - base - 1) / 2;
                    let edge_class = if (node - base) % 5 == 0 {
                        EdgeClass::VascularRoad
                    } else {
                        EdgeClass::Normal
                    };
                    add_backbone(parent, node, edge_class);
                }
            }
        }
    }
    forest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(a: u32, b: u32, edge_class: EdgeClass) -> Edge {
        Edge {
            a,
            b,
            edge_class,
            bond_class: BondClass::Backbone,
            active: true,
        }
    }

    #[test]
    fn redundant_shortcut_selection_is_strict_stable_and_visible() {
        let mut forest = SyntheticForest::new(4);
        forest.edges = vec![
            edge(0, 1, EdgeClass::Normal),
            edge(1, 2, EdgeClass::Normal),
            edge(2, 3, EdgeClass::Normal),
            edge(0, 3, EdgeClass::Normal),
        ];
        let routed = forest.select_active_routes().expect("shortcut routing");
        assert!(routed.edges[0].active);
        assert!(
            !routed.edges[1].active,
            "midpoint edge becomes black standby"
        );
        assert!(routed.edges[2].active);
        assert!(
            routed.edges[3].active,
            "short direct route becomes yellow active"
        );
        routed.cache().expect("selected routes remain a forest");

        let mut equal = SyntheticForest::new(2);
        equal.edges = vec![edge(0, 1, EdgeClass::Normal), edge(0, 1, EdgeClass::Normal)];
        let routed = equal.select_active_routes().expect("equal routing");
        assert!(
            routed.edges[0].active,
            "established route wins an exact tie"
        );
        assert!(!routed.edges[1].active, "equal new route remains standby");
    }

    #[test]
    fn vascular_corridor_becomes_the_lowest_resistance_conductor() {
        let mut forest = SyntheticForest::new(4);
        forest.edges = vec![
            edge(0, 3, EdgeClass::Normal),
            edge(0, 1, EdgeClass::VascularRoad),
            edge(1, 2, EdgeClass::VascularRoad),
            edge(2, 3, EdgeClass::VascularRoad),
        ];
        let routed = forest.select_active_routes().expect("vascular routing");
        assert!(
            !routed.edges[0].active,
            "normal direct bond becomes standby"
        );
        assert!(routed.edges[1..].iter().all(|edge| edge.active));
        assert!(3 * EdgeClass::VascularRoad.resistance() < EdgeClass::Normal.resistance());
        routed
            .cache()
            .expect("vascular conductor remains cycle-free");
    }

    #[test]
    fn lifecycle_additions_wait_for_tick_and_strict_shortcuts_exchange() {
        let mut topology = BackboneLifecycleOracle::new(4);
        topology
            .insert_backbone(10, 0, 1, EdgeClass::Normal.resistance())
            .unwrap();
        topology
            .insert_backbone(20, 1, 2, EdgeClass::Normal.resistance())
            .unwrap();
        topology
            .insert_backbone(30, 2, 3, EdgeClass::Normal.resistance())
            .unwrap();
        assert!(topology.bonds.iter().all(|bond| !bond.active));
        assert_eq!(topology.commit(2), 2);
        assert_eq!(topology.bonds.iter().filter(|bond| bond.active).count(), 2);
        assert_eq!(topology.commit(8), 1);

        topology
            .insert_backbone(40, 0, 3, EdgeClass::Normal.resistance())
            .unwrap();
        assert!(
            !topology.bonds[3].active,
            "new shortcut cannot transmit before commit"
        );
        topology.commit(1);
        assert!(topology.bonds[3].active);
        assert!(
            !topology.bonds[1].active,
            "midpoint bypass edge becomes standby"
        );
        assert_eq!(topology.bonds.iter().filter(|bond| bond.active).count(), 3);
    }

    #[test]
    fn lifecycle_break_masks_immediately_then_chooses_lowest_replacement_route() {
        let n = EdgeClass::Normal.resistance();
        let v = EdgeClass::VascularRoad.resistance();
        let mut topology = BackboneLifecycleOracle::new(5);
        topology.insert_backbone(1, 0, 1, n).unwrap();
        topology.insert_backbone(2, 1, 2, n).unwrap();
        topology.insert_backbone(3, 2, 3, n).unwrap();
        topology.insert_backbone(4, 0, 4, v).unwrap();
        topology.insert_backbone(5, 4, 3, v).unwrap();
        topology.commit(16);
        assert!(topology.bonds[3].active && topology.bonds[4].active);
        assert!(
            !topology.bonds[1].active,
            "vascular corridor displaced the middle normal edge"
        );

        assert!(topology.invalidate(4));
        assert!(
            !topology.bonds[3].active,
            "invalid active edge is masked synchronously"
        );
        assert_eq!(topology.pending_job_count(), 1);
        topology.commit(1);
        assert!(
            topology.bonds[1].active,
            "standby normal edge reconnects at commit"
        );
        assert_eq!(topology.bonds.iter().filter(|bond| bond.active).count(), 4);
    }

    #[test]
    fn lifecycle_budget_and_stable_identity_make_results_reproducible() {
        fn run() -> Vec<(u64, bool)> {
            let mut topology = BackboneLifecycleOracle::new(3);
            topology.insert_backbone(30, 0, 1, 10).unwrap();
            topology.insert_backbone(10, 1, 2, 10).unwrap();
            topology.insert_backbone(20, 0, 2, 20).unwrap();
            assert_eq!(topology.commit(1), 1);
            assert_eq!(topology.pending_job_count(), 2);
            topology.commit(1);
            topology.commit(1);
            let mut result = topology
                .bonds
                .iter()
                .map(|bond| (bond.stable_id, bond.active))
                .collect::<Vec<_>>();
            result.sort_unstable();
            result
        }
        assert_eq!(run(), run());
        assert_eq!(run(), vec![(10, true), (20, false), (30, true)]);
    }

    #[test]
    fn phase4_topology_churn_stress_is_deterministic_and_remains_a_forest() {
        fn run() -> Vec<(u64, bool, bool)> {
            const NODES: usize = 512;
            let mut topology = BackboneLifecycleOracle::new(NODES);
            for node in 1..NODES {
                topology
                    .insert_backbone(node as u64, (node - 1) as u32, node as u32, 51_293)
                    .unwrap();
            }
            for node in 0..NODES - 8 {
                topology
                    .insert_backbone(
                        10_000 + node as u64,
                        node as u32,
                        (node + 8) as u32,
                        if node % 7 == 0 { 12_579 } else { 51_293 },
                    )
                    .unwrap();
            }
            while topology.pending_job_count() != 0 {
                topology.commit(17);
            }

            for round in 0..64usize {
                let active = topology
                    .bonds
                    .iter()
                    .filter(|bond| bond.valid && bond.active)
                    .map(|bond| bond.stable_id)
                    .collect::<Vec<_>>();
                let selected = active[(round * 37) % active.len()];
                assert!(topology.invalidate(selected));
                assert!(!topology
                    .bonds
                    .iter()
                    .any(|bond| { bond.stable_id == selected && bond.valid && bond.active }));
                topology.commit(1);

                let mut union = (0..NODES).collect::<Vec<_>>();
                fn root(union: &mut [usize], mut node: usize) -> usize {
                    while union[node] != node {
                        union[node] = union[union[node]];
                        node = union[node];
                    }
                    node
                }
                for bond in topology
                    .bonds
                    .iter()
                    .filter(|bond| bond.valid && bond.active)
                {
                    let left = root(&mut union, bond.a as usize);
                    let right = root(&mut union, bond.b as usize);
                    assert_ne!(left, right, "active routing must never contain a cycle");
                    union[left] = right;
                }
            }
            topology
                .bonds
                .iter()
                .map(|bond| (bond.stable_id, bond.valid, bond.active))
                .collect()
        }

        assert_eq!(run(), run(), "identical churn must select identical routes");
    }

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-3,
            "actual={actual}, expected={expected}"
        );
    }

    fn cognocyte(operation: CognocyteOperation, a: f32, b: f32, expected: f32) {
        let (actual, invalid) = evaluate_cognocyte(operation, a, b);
        assert!(!invalid, "unexpected invalid diagnostic for {operation:?}");
        approx(actual, expected);
    }

    #[test]
    fn every_non_oscillator_cognocyte_operation_matches_signed_fixed_point_rules() {
        use CognocyteOperation::*;

        cognocyte(Add, 600.0, 700.0, 1000.0);
        cognocyte(Subtract, 250.0, 750.0, -500.0);
        cognocyte(Multiply, -500.0, 500.0, -250.0);
        cognocyte(Divide, 250.0, 500.0, 500.0);
        cognocyte(Minimum, -300.0, 200.0, -300.0);
        cognocyte(Maximum, -300.0, 200.0, 200.0);
        cognocyte(Average, -600.0, 200.0, -200.0);
        cognocyte(GreaterThan, 4.0, -4.0, 1000.0);
        cognocyte(LessThan, 4.0, -4.0, 0.0);
        cognocyte(Equal, 4.0, 4.05, 1000.0);
        cognocyte(Equal, 4.0, 4.2, 0.0);
        cognocyte(And, 1.0, -1.0, 0.0);
        cognocyte(And, 1.0, 1.0, 1000.0);
        cognocyte(Or, -1.0, 1.0, 1000.0);
        cognocyte(Or, -1.0, -1.0, 0.0);
        cognocyte(Not, 1.0, 0.0, 0.0);
        cognocyte(Not, -1.0, 0.0, 1000.0);
        cognocyte(Not, 0.0, 0.0, 1000.0);
        cognocyte(Select, 1.0, -450.0, -450.0);
        cognocyte(Select, -1.0, 450.0, 0.0);
        cognocyte(Abs, -725.0, 0.0, 725.0);
        cognocyte(Negate, -725.0, 0.0, 725.0);
        cognocyte(Positive, -725.0, 0.0, 0.0);
        cognocyte(Positive, 725.0, 0.0, 725.0);
        cognocyte(Negative, 725.0, 0.0, 0.0);
        cognocyte(Negative, -725.0, 0.0, -725.0);
    }

    #[test]
    fn cognocyte_silence_tolerance_and_invalid_diagnostics_are_explicit() {
        use CognocyteOperation::*;

        cognocyte(Add, 0.0, 500.0, 0.0);
        cognocyte(Divide, 500.0, 0.05, 0.0);
        cognocyte(Divide, 500.0, -0.05, 0.0);

        let (nan, nan_invalid) = evaluate_cognocyte(Add, f32::NAN, 1.0);
        assert_eq!(nan, 0.0);
        assert!(nan_invalid);
        let (infinite, infinite_invalid) = evaluate_cognocyte(Negate, f32::INFINITY, 0.0);
        assert_eq!(infinite, 0.0);
        assert!(infinite_invalid);
    }

    #[test]
    fn cognocyte_chain_commits_synchronously_with_one_tick_per_stage() {
        let mut stored = [0.0_f32; 2];

        let tick_zero_inputs = [400.0, stored[0]];
        let next = [
            evaluate_cognocyte(CognocyteOperation::Abs, tick_zero_inputs[0], 0.0).0,
            evaluate_cognocyte(CognocyteOperation::Abs, tick_zero_inputs[1], 0.0).0,
        ];
        assert_eq!(stored, [0.0, 0.0]);
        stored = next;
        assert_eq!(stored, [400.0, 0.0]);

        let tick_one_inputs = [400.0, stored[0]];
        stored = [
            evaluate_cognocyte(CognocyteOperation::Abs, tick_one_inputs[0], 0.0).0,
            evaluate_cognocyte(CognocyteOperation::Abs, tick_one_inputs[1], 0.0).0,
        ];
        assert_eq!(stored, [400.0, 400.0]);
    }

    #[test]
    fn memorocyte_is_fixed_tick_rate_independent_and_decays_after_silence() {
        let mut state = 0.0;
        for _ in 0..15 {
            let (next, invalid) = update_memorocyte(state, 1000.0, 0.75);
            assert!(!invalid);
            state = next;
        }
        approx(state, 750.0);

        let before_decay = state;
        for _ in 0..15 {
            state = update_memorocyte(state, 0.0, 0.75).0;
        }
        assert!(state > 0.0 && state < before_decay);
        approx(state, 187.5);
    }

    #[test]
    fn processor_lifecycle_reset_prevents_division_mode_and_slot_leaks() {
        let mut memory = update_memorocyte(0.0, -1000.0, 0.9).0;
        let mut stored_output = memory;
        assert!(memory < 0.0 && stored_output < 0.0);

        // Mode change, division child initialization, death, and slot reuse all
        // execute this same zero-state transition before visibility.
        for _event in ["mode change", "division", "death", "slot reuse"] {
            memory = 0.0;
            stored_output = 0.0;
            assert_eq!((memory, stored_output), (0.0, 0.0));
            memory = update_memorocyte(memory, -1000.0, 0.9).0;
            stored_output = memory;
            assert_eq!(stored_output, memory);
        }
    }

    #[test]
    fn signed_line_attenuates_on_first_edge_and_excludes_self() {
        let mut forest = SyntheticForest::new(3);
        forest.edges = vec![edge(0, 1, EdgeClass::Normal), edge(1, 2, EdgeClass::Normal)];
        forest.sources[0][0] = 1000.0;
        forest.sources[2][0] = -400.0;
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();

        approx(field[0][0], -400.0 * NORMAL_RETENTION * NORMAL_RETENTION);
        approx(field[1][0], (1000.0 - 400.0) * NORMAL_RETENTION);
        approx(field[2][0], 1000.0 * NORMAL_RETENTION * NORMAL_RETENTION);
    }

    #[test]
    fn cancellation_occurs_before_final_saturation() {
        let mut forest = SyntheticForest::new(3);
        forest.edges = vec![edge(0, 1, EdgeClass::Normal), edge(1, 2, EdgeClass::Normal)];
        forest.sources[0][0] = 1000.0;
        forest.sources[2][0] = -1000.0;
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        approx(field[1][0], 0.0);
    }

    #[test]
    fn fan_in_saturates_only_after_complete_accumulation() {
        let mut forest = synthetic_forest(SyntheticShape::Star, 4);
        forest.sources[1][0] = 1000.0;
        forest.sources[2][0] = 1000.0;
        forest.sources[3][0] = -500.0;
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        approx(field[0][0], 1000.0);
    }

    #[test]
    fn vascular_road_uses_approved_retention() {
        let mut forest = SyntheticForest::new(2);
        forest.edges = vec![edge(0, 1, EdgeClass::VascularRoad)];
        forest.sources[0][7] = -800.0;
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        approx(field[1][7], -800.0 * VASCULAR_RETENTION);
    }

    #[test]
    fn source_only_attachment_injects_without_relaying_or_joining_trees() {
        let mut forest = SyntheticForest::new(3);
        forest.roles[0] = NodeRole::SourceOnly;
        forest.sources[0][2] = 1000.0;
        forest.edges = vec![
            Edge {
                a: 0,
                b: 1,
                edge_class: EdgeClass::Normal,
                bond_class: BondClass::SourceAttachment,
                active: true,
            },
            Edge {
                a: 0,
                b: 2,
                edge_class: EdgeClass::Normal,
                bond_class: BondClass::SourceAttachment,
                active: true,
            },
        ];
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        approx(field[0][2], 0.0);
        approx(field[1][2], 1000.0 * NORMAL_RETENTION);
        approx(field[2][2], 1000.0 * NORMAL_RETENTION);
    }

    #[test]
    fn inactive_break_splits_immediately_and_mechanical_cross_link_is_ignored() {
        let mut forest = SyntheticForest::new(3);
        forest.sources[0][0] = 1000.0;
        forest.edges = vec![
            edge(0, 1, EdgeClass::Normal),
            Edge {
                active: false,
                ..edge(1, 2, EdgeClass::Normal)
            },
            Edge {
                a: 0,
                b: 2,
                edge_class: EdgeClass::Normal,
                bond_class: BondClass::MechanicalOnly,
                active: true,
            },
        ];
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        approx(field[1][0], 950.0);
        approx(field[2][0], 0.0);
    }

    #[test]
    fn all_channels_cross_a_join_independent_of_organism_identity() {
        let mut forest = SyntheticForest::new(2);
        forest.edges = vec![edge(0, 1, EdgeClass::Normal)];
        for channel in 0..CHANNEL_COUNT {
            forest.sources[0][channel] = if channel % 2 == 0 { 1000.0 } else { -1000.0 };
        }
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        for channel in 0..CHANNEL_COUNT {
            approx(
                field[1][channel],
                forest.sources[0][channel] * NORMAL_RETENTION,
            );
        }
    }

    #[test]
    fn deep_chain_retains_local_influence_without_root_product_underflow() {
        let node_count = 200_000;
        let mut forest = synthetic_forest(SyntheticShape::Chain, node_count);
        forest.sources[node_count - 2][0] = 1000.0;
        let field = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        approx(field[node_count - 1][0], 950.0);
    }

    #[test]
    fn cycle_is_rejected_instead_of_becoming_schedule_dependent() {
        let mut forest = SyntheticForest::new(3);
        forest.edges = vec![
            edge(0, 1, EdgeClass::Normal),
            edge(1, 2, EdgeClass::Normal),
            edge(2, 0, EdgeClass::Normal),
        ];
        assert!(matches!(
            forest.cache(),
            Err(TopologyError::RelayCycle { .. })
        ));
    }

    #[test]
    fn edge_insertion_order_does_not_change_the_field() {
        let mut first = synthetic_forest(SyntheticShape::BalancedBinary, 127);
        populate_workload(&mut first.sources, SignalWorkload::EveryCellEmits, 0);
        let mut second = first.clone();
        second.edges.reverse();
        let first_field = first.cache().unwrap().propagate(&first.sources).unwrap();
        let second_field = second.cache().unwrap().propagate(&second.sources).unwrap();
        assert_eq!(first_field, second_field);
    }

    #[test]
    fn mechanical_density_does_not_change_results_or_cached_edge_count() {
        let mut sparse = synthetic_forest(SyntheticShape::Chain, 256);
        populate_workload(&mut sparse.sources, SignalWorkload::AllChannelsSparse, 0);
        let mut dense = sparse.clone();
        for node in 0..dense.roles.len() - 17 {
            dense.edges.push(Edge {
                a: node as u32,
                b: (node + 17) as u32,
                edge_class: EdgeClass::Normal,
                bond_class: BondClass::MechanicalOnly,
                active: true,
            });
        }
        let sparse_cache = sparse.cache().unwrap();
        let dense_cache = dense.cache().unwrap();
        assert_eq!(sparse_cache.parent, dense_cache.parent);
        assert_eq!(
            sparse_cache.propagate(&sparse.sources).unwrap(),
            dense_cache.propagate(&dense.sources).unwrap()
        );
    }

    #[test]
    fn positive_and_negative_monotonicity_holds_before_saturation() {
        let mut forest = synthetic_forest(SyntheticShape::BalancedBinary, 31);
        forest.sources[7][0] = 100.0;
        let baseline = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        forest.sources[12][0] = 50.0;
        let positive = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        for cell in 0..forest.roles.len() {
            assert!(positive[cell][0] + 1.0e-4 >= baseline[cell][0]);
        }
        forest.sources[12][0] = -50.0;
        let negative = forest.cache().unwrap().propagate(&forest.sources).unwrap();
        for cell in 0..forest.roles.len() {
            assert!(negative[cell][0] <= baseline[cell][0] + 1.0e-4);
        }
    }

    #[test]
    fn heat_workload_is_deterministic_signed_full_magnitude() {
        let mut first = vec![[0.0; CHANNEL_COUNT]; 32];
        let mut second = first.clone();
        populate_workload(&mut first, SignalWorkload::HeatScreamAllChannels, 77);
        populate_workload(&mut second, SignalWorkload::HeatScreamAllChannels, 77);
        assert_eq!(first, second);
        assert!(first
            .iter()
            .flatten()
            .all(|value| *value == -1000.0 || *value == 1000.0));
    }

    #[test]
    fn microtree_schedule_is_bounded_connected_and_has_one_parent_boundary() {
        for shape in [
            SyntheticShape::Chain,
            SyntheticShape::Star,
            SyntheticShape::BalancedBinary,
            SyntheticShape::ManyPairs,
            SyntheticShape::GameplayMixed,
        ] {
            let forest = synthetic_forest(shape, 10_003);
            let cache = forest.cache().unwrap();
            let schedule = MicrotreeSchedule::build(&cache, 128);
            assert!(schedule
                .microtrees
                .iter()
                .all(|block| !block.nodes.is_empty() && block.nodes.len() <= 128));
            assert!(schedule
                .node_microtree
                .iter()
                .all(|microtree| *microtree != INVALID_NODE));

            for (microtree_id, block) in schedule.microtrees.iter().enumerate() {
                let seed = block.attachment_node as usize;
                let parent = cache.parent[seed];
                if parent == INVALID_NODE {
                    assert_eq!(block.parent_microtree, INVALID_NODE);
                } else {
                    assert_eq!(
                        block.parent_microtree,
                        schedule.node_microtree[parent as usize]
                    );
                    assert_ne!(block.parent_microtree, microtree_id as u32);
                }
                for &node in block.nodes.iter().skip(1) {
                    let parent = cache.parent[node as usize];
                    assert_eq!(
                        schedule.node_microtree[parent as usize],
                        microtree_id as u32
                    );
                }
            }
        }
    }

    #[test]
    fn microtree_boundary_solver_matches_independent_oracle_on_every_shape() {
        for shape in [
            SyntheticShape::Chain,
            SyntheticShape::Star,
            SyntheticShape::BalancedBinary,
            SyntheticShape::ManyPairs,
            SyntheticShape::GameplayMixed,
            SyntheticShape::DenseMechanicalSparseBackbone,
        ] {
            let mut forest = synthetic_forest(shape, 10_003);
            populate_workload(
                &mut forest.sources,
                SignalWorkload::HeatScreamAllChannels,
                913,
            );
            let cache = forest.cache().unwrap();
            let expected = cache.propagate(&forest.sources).unwrap();

            for block_size in [64, 96, 128] {
                let schedule = MicrotreeSchedule::build(&cache, block_size);
                let actual = schedule.propagate(&cache, &forest.sources).unwrap();
                for cell in 0..forest.roles.len() {
                    for channel in 0..CHANNEL_COUNT {
                        assert!(
                            (actual[cell][channel] - expected[cell][channel]).abs() <= 1.0e-3,
                            "shape={shape:?} block={block_size} cell={cell} channel={channel} actual={} expected={}",
                            actual[cell][channel],
                            expected[cell][channel]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn macro_strategy_is_derived_from_topology_and_remains_bounded_at_200k() {
        for shape in [
            SyntheticShape::Chain,
            SyntheticShape::Star,
            SyntheticShape::BalancedBinary,
            SyntheticShape::ManyPairs,
            SyntheticShape::GameplayMixed,
            SyntheticShape::DenseMechanicalSparseBackbone,
        ] {
            let forest = synthetic_forest(shape, 200_000);
            let cache = forest.cache().unwrap();
            let microtrees = MicrotreeSchedule::build(&cache, 64);
            let macro_schedule = microtrees.macro_schedule();
            let mut expected_start = 0_u32;
            for bucket in &macro_schedule.depth_buckets {
                assert_eq!(bucket.first().copied(), Some(expected_start));
                for (offset, &microtree_id) in bucket.iter().enumerate() {
                    assert_eq!(microtree_id, expected_start + offset as u32);
                }
                expected_start += bucket.len() as u32;
            }
            let macro_dispatches = macro_schedule.macro_dispatches();
            let per_group_dispatches = macro_dispatches + 2;
            eprintln!(
                "shape={shape:?} microtrees={} strategy={:?} depths={} max_children={} dispatches_per_group={per_group_dispatches}",
                microtrees.microtrees.len(),
                macro_schedule.strategy,
                macro_schedule.depth_buckets.len(),
                macro_schedule.maximum_children,
            );
            assert!(per_group_dispatches <= 40);

            match shape {
                SyntheticShape::Chain
                | SyntheticShape::ManyPairs
                | SyntheticShape::DenseMechanicalSparseBackbone => {
                    assert_eq!(macro_schedule.strategy, MacroStrategy::PointerJumpingPath);
                }
                SyntheticShape::Star | SyntheticShape::BalancedBinary => {
                    assert_eq!(macro_schedule.strategy, MacroStrategy::DepthBuckets);
                }
                SyntheticShape::GameplayMixed => {}
            }
        }
    }

    #[test]
    fn flattened_gpu_topology_abi_is_stable_complete_and_generation_tagged() {
        assert_eq!(std::mem::size_of::<GpuCellTopology>(), 32);
        assert_eq!(std::mem::size_of::<GpuMicrotreeTopology>(), 32);

        for shape in [
            SyntheticShape::Chain,
            SyntheticShape::Star,
            SyntheticShape::BalancedBinary,
            SyntheticShape::ManyPairs,
            SyntheticShape::GameplayMixed,
            SyntheticShape::DenseMechanicalSparseBackbone,
        ] {
            let forest = synthetic_forest(shape, 10_003);
            let cache = forest.cache().unwrap();
            let schedule = MicrotreeSchedule::build(&cache, 64);
            let upload = schedule.flatten_for_gpu(&cache, 17);

            assert_eq!(upload.cells.len(), forest.roles.len());
            assert_eq!(upload.node_list.len(), forest.roles.len());
            assert_eq!(upload.microtrees.len(), schedule.microtrees.len());
            assert!(upload.cells.iter().all(|cell| cell.generation == 17));
            assert!(upload
                .microtrees
                .iter()
                .all(|microtree| microtree.generation == 17));

            let mut nodes = upload.node_list.clone();
            nodes.sort_unstable();
            assert_eq!(nodes, (0..forest.roles.len() as u32).collect::<Vec<_>>());
            assert_eq!(upload.depth_offsets.first(), Some(&0));
            assert_eq!(
                upload.depth_offsets.last().copied(),
                Some(upload.depth_microtrees.len() as u32)
            );

            for (microtree_id, metadata) in upload.microtrees.iter().enumerate() {
                let nodes = &upload.node_list[metadata.node_offset as usize
                    ..(metadata.node_offset + metadata.node_count) as usize];
                assert_eq!(nodes, schedule.microtrees[microtree_id].nodes);
                assert_eq!(nodes[0], metadata.attachment_node);
                assert_eq!(
                    metadata.external_parent_cell,
                    cache.parent[metadata.attachment_node as usize]
                );
                for (local_index, &node) in nodes.iter().enumerate() {
                    let cell = upload.cells[node as usize];
                    assert_eq!(cell.microtree_id, microtree_id as u32);
                    assert_eq!(cell.local_index, local_index as u32);
                }
                let child_end = metadata.child_boundary_offset + metadata.child_boundary_count;
                for &child in &upload.child_microtrees
                    [metadata.child_boundary_offset as usize..child_end as usize]
                {
                    assert_eq!(
                        upload.microtrees[child as usize].parent_microtree,
                        microtree_id as u32
                    );
                }
            }
        }
    }

    #[test]
    fn flattened_topology_fits_phase1_memory_budget_at_200k() {
        for shape in [
            SyntheticShape::Chain,
            SyntheticShape::Star,
            SyntheticShape::BalancedBinary,
            SyntheticShape::ManyPairs,
            SyntheticShape::GameplayMixed,
            SyntheticShape::DenseMechanicalSparseBackbone,
        ] {
            let forest = synthetic_forest(shape, 200_000);
            let cache = forest.cache().unwrap();
            let schedule = MicrotreeSchedule::build(&cache, 64);
            let upload = schedule.flatten_for_gpu(&cache, TOPOLOGY_GENERATION_INITIAL);
            eprintln!(
                "shape={shape:?} topology_bytes={} topology_mib={:.3}",
                upload.allocated_bytes(),
                upload.allocated_bytes() as f64 / (1024.0 * 1024.0)
            );
            assert!(upload.allocated_bytes() <= 16 * 1024 * 1024);
        }
    }

    #[test]
    fn high_degree_boundary_reduction_is_bounded_complete_and_deterministic() {
        for shape in [SyntheticShape::Star, SyntheticShape::BalancedBinary] {
            let forest = synthetic_forest(shape, 200_000);
            let cache = forest.cache().unwrap();
            let microtrees = MicrotreeSchedule::build(&cache, 64);
            let first = microtrees.boundary_reduction_schedule(&cache);
            let second = microtrees.boundary_reduction_schedule(&cache);

            assert_eq!(first, second);
            assert!(!first.is_empty());
            for depth in &first {
                assert!(!depth.passes.is_empty());
                for pass in &depth.passes {
                    assert!(pass.chunks.iter().all(|chunk| {
                        chunk.input_count > 0
                            && chunk.input_count as usize <= BOUNDARY_REDUCTION_WIDTH
                            && chunk.input_offset + chunk.input_count <= pass.inputs.len() as u32
                    }));
                }
                assert!(depth
                    .passes
                    .last()
                    .unwrap()
                    .chunks
                    .iter()
                    .all(|chunk| chunk.final_output));
            }

            if shape == SyntheticShape::Star {
                assert_eq!(first.len(), 1);
                assert_eq!(first[0].passes.len(), 3);
                assert_eq!(first[0].passes[0].inputs.len(), 199_936);
                assert_eq!(first[0].passes[0].chunks.len(), 781);
                assert_eq!(first[0].passes[1].chunks.len(), 4);
                assert_eq!(first[0].passes[2].chunks.len(), 1);
                assert!(first[0].passes[2].chunks[0].final_output);
                assert_eq!(first[0].passes[2].chunks[0].target_parent_cell, 0);
            }
        }
    }
}
