from numpy.typing import NDArray
import numpy as np
from wgpu import GPUBindGroupLayout, GPURenderPassEncoder, GPUTexture
import struct

from wgut.core import get_device

SHADER = """
struct DATA {
    albedo_color: vec3<f32>,
    albedo_tex: i32,
    emissivity_color: vec3<f32>,
    emissivity_tex: i32,
    normal_value: vec3<f32>,
    normal_tex: i32,
    roughness_value: f32,
    roughness_tex: i32,
    metalicity_value: f32,
    metalicity_tex: i32,
    occlusion_tex: i32,
    // implicit padding of 12 bytes
}

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
    let id = i32(in.mat_id);

    let albedo = sample(data[mat_id].albedo_tex, uv) * data[mat_id].albedo_color;
    let emissivity = sample(data[mat_id].emissivity_tex, uv) * data[mat_id].emissivity_color;
    let normal = 2.0 * sample(data[mat_id].normal_tex, uv) * data[mat_id].normal_value - 1.0;
    let roughness = sample(data[mat_id].roughness_tex, uv).x * data[mat_id].roughness_value;
    let metalicity = sample(data[mat_id].metalicity_tex, uv).x * data[mat_id].metalicity_value;
    let occlusion = sample(data[mat_id].occlusion_tex, uv);

    let F0 = vec3<f32>(0.15); // Base Reflectance
    let alpha = roughness * roughness;

    let N = normalize(normal.x * in.tangent + normal.y * in.bitangent + normal.z * in.normal);
    let V = normalize(camera.position.xyz - in.position.xyz);

    var color = vec3<f32>(0.0);

    for(var i: i32 = 0; i < light_count; i++) {
        let lightColor = lights[i].color.rgb * lights[i].color.a;
        // ambiant light
        if(lights[i].position.x == 0.0 && lights[i].position.y == 0.0 && lights[i].position.z == 0.0 && lights[i].position.w == 0.0) {
            color += albedo * lightColor * occlusion;
        }
        else {
            var L = normalize(-lights[i].position.xyz);  // direction light
            if(lights[i].position.w != 0.0) {
                L = normalize(lights[i].position.xyz - in.position.xyz); // point light
            }
            let H = normalize(V + L);

            color += PBR(alpha, F0, N, L, V, H, albedo, lightColor, emissivity, metalicity);
        }
    }

    return vec4<f32>(color, 1.0);
}
"""


class PbrMaterial:
    def __init__(self):
        pass

    @staticmethod
    def get_fragment() -> str:
        return SHADER

    @staticmethod
    def get_bind_group_layouts() -> list[GPUBindGroupLayout]:
        # To add bind groups
        return []

    @staticmethod
    def get_data_size() -> int:
        return 5 * 16  # data size with padding

    def get_texture_count(self) -> int:
        return 0

    def get_data(self, tex_ids: list[int]) -> bytes:
        # must send the textures to the renderer to get the texture's ids and put them in the data
        return b""

    def get_textures(self) -> list[GPUTexture]:
        return []

    @staticmethod
    def set_bindings(render_pass: GPURenderPassEncoder, bind_group_offset=0):
        # probably noting to do here but can be usefull if additional bindings is used
        pass
