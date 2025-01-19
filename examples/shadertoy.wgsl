fn mainImage(fragCoord: vec2<f32>) -> vec4<f32> {
    let uv = fragCoord / iResolution.xy;
    let col = vec3<f32>(uv, 0.0);
    return vec4<f32>(col, 1.0);
}
