@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) bitangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = camera * in.position;
    out.color = in.color;
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var lightDir = vec3<f32>(1.0, 1.0, -1.0);
    var shading = clamp(dot(normalize(in.normal), normalize(lightDir)), 0.0, 1.0);
    var color = vec3<f32>(1.0, 0.0, 0.0);
    return vec4<f32>(shading * color, 1.0);
}
