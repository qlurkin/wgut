@group(0) @binding(0)
var<storage,read> data1: array<i32>;

@group(0) @binding(1)
var<storage,read_write> data2: array<i32>;

@compute
@workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x + index.y * Y_STRIDE;
    data2[i] = data1[i] * data1[i];
}

