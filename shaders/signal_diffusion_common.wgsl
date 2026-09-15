struct Params {
    count: u32, tick: u32, degree: u32, resolution: u32,
    dt: f32, conductance: f32, retention: f32, production_scale: f32,
    time: f32, radius: f32, grid_cell: f32, padding: f32,
    grid_origin: vec4<f32>,
}
struct CellState {
    identity: u32, mode: u32, config_hash: u32, live: u32,
    memory: f32, output: f32, channel: u32, padding: u32,
}
struct Mode {
    photo: vec4<f32>, lipo: vec4<f32>, values: vec4<f32>,
    processor: vec4<u32>, oscillator: vec4<f32>, light_filter: vec4<f32>,
}
