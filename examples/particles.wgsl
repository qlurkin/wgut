@group(0) @binding(0)
var<storage,read_write> positions: array<vec4<f32>>;

@compute
@workgroup_size(4)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x;
    positions[i] = positions[i] + vec4<f32>(0.0, 0.01, 0.0, 0.0);
}

