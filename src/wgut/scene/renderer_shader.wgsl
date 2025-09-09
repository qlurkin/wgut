struct Camera {
    matrix: mat4x4<f32>,
    position: vec4<f32>,
};

// 4th component of color is intensity, 4th component of position equal to 0 means Direction Light.
// All four component of position to 0 means ambiant light.
struct Light {
    position: vec4<f32>,
    color: vec4<f32>,
}

struct Material {
    albedo_color: vec4<f32>,
    emissivity_color: vec3<f32>,
    emissivity_tex: i32,
    normal_value: vec3<f32>,
    normal_tex: i32,
    roughness_value: f32,
    roughness_tex: i32,
    metalicity_value: f32,
    metalicity_tex: i32,
    occlusion_tex: i32,
    albedo_tex: i32,
    padding1: f32,  // should not be mandatory but needed on osX
    padding2: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var<uniform> light_count: i32;
@group(0) @binding(3) var<storage, read> materials: array<Material>;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) bitangent: vec3<f32>,
    @location(6) mat_id: f32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) bitangent: vec3<f32>,
    @location(6) mat_id: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = camera.matrix * in.position;
    out.position = in.position;
    out.color = in.color;
    out.uv = in.uv;
    out.normal = in.normal;
    out.tangent = in.tangent;
    out.bitangent = in.bitangent;
    out.mat_id = in.mat_id;
    return out;
}

// TERTURES_INJECTION_START
fn sample(tex_id: i32, uv: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(1.0);
}
// TEXTURE_INJECTION_END

const PI: f32 = 3.141592;

fn transform(m: mat4x4<f32>, v: vec3<f32>) -> vec3<f32> {
    let v4 = vec4(v, 1.0);
    let res4 = m * v4;
    return res4.xyz / res4.w;
}

fn transform3x3(m: mat4x4<f32>, v: vec3<f32>) -> vec3<f32> {
    let v4 = vec4(v, 0.0);
    return (m * v4).xyz;
}

// https://www.youtube.com/watch?v=RRE-F57fbXw

// GGX/Trowbridge-Reitz Normal Distribution Function
fn D(alpha: f32, N: vec3<f32>, H: vec3<f32>) -> f32 {
    let numerator = pow(alpha, 2.0);

    let NdotH = max(dot(N, H), 0.0);
    var denominator = PI * pow(pow(NdotH, 2.0) * (numerator - 1.0) + 1.0, 2.0);
    denominator = max(denominator, 0.000001);

    return numerator / denominator;
}

// Schilick-Beckmann Geometry Shadowing Function
fn G1(alpha: f32, N: vec3<f32>, X: vec3<f32>) -> f32 {
    let numerator = max(dot(N, X), 0.0);

    let k = alpha / 2.0;
    var denominator = numerator * (1.0 - k) + k;
    denominator = max(denominator, 0.000001);

    return numerator / denominator;
}

// Smith Model
fn G(alpha: f32, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>) -> f32 {
    return G1(alpha, N, V) * G1(alpha, N, L);
}

// Fresnel-Schlick Function
fn F(F0: vec3<f32>, V: vec3<f32>, H: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
}

// Rendering Equation for one light source
fn PBR(alpha: f32, F0: vec3<f32>, N: vec3<f32>, L: vec3<f32>, V: vec3<f32>, H: vec3<f32>, albedo: vec3<f32>, lightColor: vec3<f32>, emissivity: vec3<f32>, metallic: f32) -> vec3<f32> {
    let Ks = F(F0, V, H);
    let Kd = (vec3<f32>(1.0) - Ks) * (1.0 - metallic);

    let lambert = albedo / PI;

    let cookTorranceNumerator = D(alpha, N, H) * G(alpha, N, V, L);
    var cookTorranceDenominator = 4.0 * max(dot(V, N), 0.0) * max(dot(L, N), 0.0);
    cookTorranceDenominator = max(cookTorranceDenominator, 0.000001);
    let cookTorrance = cookTorranceNumerator / cookTorranceDenominator;

    let BRDF = Kd * lambert + Ks * cookTorrance;
    let outgoingLight = emissivity + BRDF * lightColor * max(dot(L, N), 0.0);

    return outgoingLight;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat_id = i32(in.mat_id);

    var albedo = sample(materials[mat_id].albedo_tex, in.uv) * materials[mat_id].albedo_color;
    albedo = pow(albedo, vec4<f32>(2.2, 2.2, 2.2, 1.0)); // gamma correction
    let emissivity = sample(materials[mat_id].emissivity_tex, in.uv).rgb * materials[mat_id].emissivity_color;
    let normal = 2.0 * sample(materials[mat_id].normal_tex, in.uv).rgb * materials[mat_id].normal_value - 1.0;
    let roughness = sample(materials[mat_id].roughness_tex, in.uv).x * materials[mat_id].roughness_value;
    let metalicity = sample(materials[mat_id].metalicity_tex, in.uv).x * materials[mat_id].metalicity_value;
    let occlusion = sample(materials[mat_id].occlusion_tex, in.uv).x;

    let F0 = vec3<f32>(0.15); // Base Reflectance
    let alpha = roughness * roughness;

    let N = normalize(normal.x * in.tangent + normal.y * in.bitangent + normal.z * in.normal);
    let V = normalize(camera.position.xyz - in.position.xyz);

    var color = vec3<f32>(0.0);

    for (var i: i32 = 0; i < light_count; i++) {
        let lightColor = lights[i].color.rgb * lights[i].color.a;
        // ambiant light
        if lights[i].position.x == 0.0 && lights[i].position.y == 0.0 && lights[i].position.z == 0.0 && lights[i].position.w == 0.0 {
            color += albedo.rgb * lightColor * occlusion;
        } else {
            var L = normalize(-lights[i].position.xyz);  // direction light
            if lights[i].position.w != 0.0 {
                L = normalize(lights[i].position.xyz - in.position.xyz); // point light
            }
            let H = normalize(V + L);

            color += PBR(alpha, F0, N, L, V, H, albedo.rgb, lightColor, emissivity, metalicity);
        }
    }

    return vec4<f32>(color, 1.0);
}
