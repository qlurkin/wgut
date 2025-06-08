@group(0) @binding(0)
var<storage,read_write> positions: array<vec4<f32>>;
@group(0) @binding(1)
var<storage,read> velocities: array<vec4<f32>>;
@group(0) @binding(2)
var<uniform> delta_time: f32;


@compute
@workgroup_size(128)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x;
    positions[i] = positions[i] + velocities[i] * delta_time;
}

