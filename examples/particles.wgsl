struct Instance {
    matrix: mat4x4<f32>,
    global_id: i32,
    pad1: i32,
    pad2: i32,
    pad3: i32,
}

@group(0) @binding(0)
var<storage,read_write> instances: array<Instance>;
@group(0) @binding(1)
var<storage,read> velocities: array<vec4<f32>>;
@group(0) @binding(2)
var<uniform> delta_time: f32;


@compute
@workgroup_size(128)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x;
    var position = instances[i].matrix[3];
    position = position + velocities[i] * delta_time;
    instances[i].matrix[3] = position;
}

