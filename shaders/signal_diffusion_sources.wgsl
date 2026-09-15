@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> current: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> production: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> states: array<CellState>;
@group(0) @binding(4) var<storage, read> death: array<u32>;
@group(0) @binding(5) var<storage, read> identities: array<u32>;
@group(0) @binding(6) var<storage, read> mode_indices: array<u32>;
@group(0) @binding(7) var<storage, read> mode_refs: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> modes: array<Mode>;
@group(0) @binding(9) var<storage, read> regulation: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read> oculocyte: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read> types: array<u32>;
@group(0) @binding(12) var<storage, read_write> nutrients: array<i32>;
@group(0) @binding(13) var<storage, read> thermal: array<u32>;
@group(0) @binding(14) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read> orientations: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read> light: array<f32>;
@group(0) @binding(17) var<storage, read> light_color: array<vec4<f32>>;
@group(0) @binding(18) var<storage, read> food: array<u32>;
@group(0) @binding(19) var<storage, read> solid: array<u32>;
@group(0) @binding(20) var<storage, read> moss: array<f32>;
@group(0) @binding(21) var<storage, read> counts: array<u32>;
// Spatial grid from the completed physics step. Sensor work scales with the
// cells encountered along the ray, independently of signal transport.
@group(0) @binding(22) var<storage, read> grid_counts: array<u32>;
@group(0) @binding(23) var<storage, read> grid_cells: array<u32>;
@group(0) @binding(24) var<uniform> spatial: vec4<f32>; // half_world,cell_size,resolution,max_per_voxel

fn voxel(p: vec3<f32>) -> u32 {
    let c = vec3<i32>(floor((p - params.grid_origin.xyz) / params.grid_cell));
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(i32(params.resolution)))) { return 0xffffffffu; }
    return u32(c.x) + u32(c.y)*params.resolution + u32(c.z)*params.resolution*params.resolution;
}
fn illumination(p: vec3<f32>) -> f32 {
    let v=voxel(p); if (v >= arrayLength(&light)) {return 0.0;} return light[v];
}
fn detected(cell: u32, config: Mode, sensor: vec4<u32>) -> bool {
    let mask=sensor.x;
    if ((mask & 16u)!=0u) {return true;}
    let origin=positions[cell].xyz;
    let q=orientations[cell];
    let forward=vec3<f32>(0.0,0.0,1.0)+2.0*cross(q.xyz,cross(q.xyz,vec3<f32>(0.0,0.0,1.0))+q.w*vec3<f32>(0.0,0.0,1.0));
    let range=clamp(bitcast<f32>(sensor.y),1.0,100.0);
    let spacing=max(min(params.grid_cell, spatial.y)*0.5,0.1);
    for (var distance=0.0; distance<=range; distance+=spacing) {
        let point=origin+forward*distance;
        let v=voxel(point);
        if ((mask & 8u)!=0u && (length(point)>=params.radius || (v<arrayLength(&solid) && solid[v]!=0u))) {return true;}
        if ((mask & 2u)!=0u && v<arrayLength(&food) && food[v]==1u) {return true;}
        if ((mask & 32u)!=0u && v<arrayLength(&moss) && moss[v]>0.0) {return true;}
        if ((mask & 4u)!=0u && illumination(point)>0.01 && v<arrayLength(&light_color)) {
            let color=light_color[v].xyz;
            if (length(color-config.light_filter.xyz)<=config.light_filter.w) {return true;}
        }
        if ((mask & 1u)!=0u) {
            let c=vec3<i32>(floor((point+vec3<f32>(spatial.x))/spatial.y));
            let res=i32(spatial.z); let maximum=u32(spatial.w);
            // Neighboring buckets cover cells whose spheres intersect the ray.
            for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
                let gc=c+vec3<i32>(x,y,z);
                if(any(gc<vec3<i32>(0))||any(gc>=vec3<i32>(res))){continue;}
                let bucket=u32(gc.x+gc.y*res+gc.z*res*res);
                if(bucket>=arrayLength(&grid_counts)){continue;}
                for(var j=0u;j<min(grid_counts[bucket],maximum);j++){
                    let other=grid_cells[bucket*maximum+j];
                    if(other>=min(params.count,counts[0])||other==cell||death[other]!=0u){continue;}
                    let rel=positions[other].xyz-origin;
                    let t=dot(rel,forward);
                    let radius=clamp(positions[other].w,0.5,2.0);
                    if(t>=0.0&&t<=range&&dot(rel,rel)-t*t<=radius*radius){return true;}
                }
            }}}
        }
    }
    return false;
}
fn request(values: ptr<function,array<vec4<f32>,4>>, channel: u32, value: f32) {
    if(channel<16u){(*values)[channel/4u][channel%4u]+=clamp(value,0.0,1000.0);}
}
@compute @workgroup_size(128)
fn sources(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell=id.x; if(cell>=params.count){return;}
    let mode=mode_indices[cell];
    let live=cell<counts[0] && death[cell]==0u && mode<arrayLength(&mode_refs);
    var state=states[cell];
    let fresh=state.live==0u||state.identity!=identities[cell];
    if(fresh||!live){for(var group=0u;group<4u;group++){current[cell*4u+group]=vec4<f32>(0.0);}}
    var config=Mode(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(0u),vec4<f32>(0.0),vec4<f32>(0.0));
    if(live){let mode_ref=u32(mode_refs[mode].z);if(mode_ref<arrayLength(&modes)){config=modes[mode_ref];}}
    let hash=config.processor.x ^ config.processor.y ^ config.processor.z ^ config.processor.w ^ bitcast<u32>(config.oscillator.x) ^ bitcast<u32>(config.oscillator.y) ^ bitcast<u32>(config.oscillator.z) ^ bitcast<u32>(config.oscillator.w) ^ types[cell];
    if(fresh||!live||state.mode!=mode||state.config_hash!=hash){state=CellState(identities[cell],mode,hash,0u,0.0,0.0,0u,0u);}
    state.live=select(0u,1u,live);
    var values: array<vec4<f32>,4>;
    if(live){
        let reg=regulation[mode];if(reg.x>=8u&&reg.x<16u){request(&values,reg.x,bitcast<f32>(reg.y));}
        if(types[cell]==7u&&detected(cell,config,oculocyte[mode])){request(&values,min(oculocyte[mode].w,7u),config.values.z);}
        if(types[cell]==3u&&config.photo.x>0.0){
            let above=illumination(positions[cell].xyz)>=config.photo.z;
            if(above != (config.photo.w==1.0)){request(&values,u32(config.photo.y),config.values.x);}
        }
        if(types[cell]==4u&&config.lipo.x>0.0){
            let above=clamp(f32(nutrients[cell])/200000.0,0.0,1.0)>=config.lipo.z;
            if(above != (config.lipo.w==1.0)){request(&values,u32(config.lipo.y),config.values.y);}
        }
        if(types[cell]==14u||types[cell]==15u){request(&values,state.channel,state.output);}
        let critical=thermal[cell]==9u;
        if(critical){for(var group=0u;group<4u;group++){values[group]=vec4<f32>(1000.0);}}
        var total=0.0;for(var group=0u;group<4u;group++){total+=dot(values[group],vec4<f32>(1.0));}
        let cost=total*0.25*params.dt*params.production_scale; // nutrients in fixed point
        let available=max(nutrients[cell],0);
        let funding=select(min(f32(available)/max(cost,0.000001),1.0),1.0,critical||cost==0.0);
        nutrients[cell]-=min(available,i32(ceil(cost*funding)));
        for(var group=0u;group<4u;group++){values[group]*=funding;}

    }
    for(var group=0u;group<4u;group++){production[cell*4u+group]=values[group];}
    states[cell]=state;
}
