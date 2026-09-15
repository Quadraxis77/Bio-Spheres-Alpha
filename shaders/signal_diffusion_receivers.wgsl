@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> states: array<CellState>;
@group(0) @binding(3) var<storage, read_write> published: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> mode_refs: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> modes: array<Mode>;
@group(0) @binding(6) var<storage, read> types: array<u32>;
fn input(cell:u32,ch:u32)->f32{return min(field[cell*4u+ch/4u][ch%4u],1000.0);}
fn evaluate(op:u32,a:f32,b:f32,c:Mode)->f32 {
    if(op==14u){
        let sine=sin((c.oscillator.x*params.time+c.oscillator.y)*6.28318530718);
        if(c.values.w==1.0){return -max(sine,0.0)*c.oscillator.z;}
        if(c.values.w==2.0){return sine*c.oscillator.z;}
        return max(sine,0.0)*c.oscillator.z;
    }
    if(op==15u){
        let phase=fract(c.oscillator.x*params.time+c.oscillator.y);
        if(c.values.w==1.0){return -phase*c.oscillator.z;}
        if(c.values.w==2.0){return (phase*2.0-1.0)*c.oscillator.z;}
        return phase*c.oscillator.z;
    }
    let unary=op==12u||(op>=16u&&op<=19u);
    if(!unary&&(a==0.0||b==0.0)){return 0.0;}
    switch op {
        case 0u:{return a+b;} case 1u:{return a-b;} case 2u:{return a*b/1000.0;}
        case 3u:{if(b<=0.1){return 0.0;}return a*1000.0/b;}
        case 4u:{return min(a,b);} case 5u:{return max(a,b);} case 6u:{return (a+b)*0.5;}
        case 7u:{return select(0.0,1000.0,a>b);} case 8u:{return select(0.0,1000.0,a<b);}
        case 9u:{return select(0.0,1000.0,abs(a-b)<=0.1);}
        case 10u:{return select(0.0,1000.0,a>0.0&&b>0.0);} case 11u:{return select(0.0,1000.0,a>0.0||b>0.0);}
        case 12u:{return select(1000.0,0.0,a>0.0);} case 13u:{return select(0.0,b,a>0.0);}
        case 16u,18u:{return a;} default:{return 0.0;}
    }
}
@compute @workgroup_size(128)
fn receivers(@builtin(global_invocation_id) id:vec3<u32>){
    let cell=id.x;if(cell>=params.count){return;}
    var state=states[cell];
    for(var group=0u;group<4u;group++){
        let value=field[cell*4u+group];
        published[cell*4u+group]=vec4<u32>(round(min(value,vec4<f32>(1000.0))));
    }
    if(state.live==0u){return;}
    let mode_ref=u32(mode_refs[state.mode].z);
    if(mode_ref>=arrayLength(&modes)){return;}
    let config=modes[mode_ref];
    let a=input(cell,min(config.processor.y,15u));
    let b=input(cell,min(config.processor.z,15u));
    state.channel=min(config.processor.w,15u);
    state.output=0.0;
    if(types[cell]==14u){state.output=clamp(evaluate(config.processor.x,a,b,config),0.0,1000.0);}
    if(types[cell]==15u){
        let rate=1.0-pow(1.0-clamp(config.oscillator.w,0.0,1.0),params.dt);
        state.memory+=(a-state.memory)*rate;
        state.output=state.memory;
    }
    states[cell]=state;
}
