struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) tex_id: f32,
};

struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) tex_coord: vec2<f32>,
  @location(1) tex_id: f32,
};

@group(0) @binding(0)
var textures: texture_2d_array<f32>;
@group(0) @binding(1)
var samplr: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coord = in.tex_coord;
    out.tex_id = in.tex_id;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(textures, samplr, in.tex_coord, i32(in.tex_id));
}
