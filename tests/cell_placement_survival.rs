#[test]
fn crowded_collision_buckets_do_not_mark_cells_dead() {
    let source = include_str!("../shaders/collision_detection.wgsl");

    assert!(
        !source.contains("cull_overcrowded_overflow_cell")
            && !source.contains("death_flags[cell_idx] = 1u"),
        "collision overflow must not kill newly placed, unbonded cells"
    );

    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("collision shader must validate");
}
