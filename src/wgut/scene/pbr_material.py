from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from wgpu import GPUBuffer, GPUTexture
from wgut.auto_render_pipeline import AutoRenderPipeline


class PbrMaterial:
    def __init__(
        self,
        albedo: str,
        normal: str,
        roughness: str,
        metalicity: str,
        emissivity: str,
        occlusion: str,
    ):
        self.__albedo = albedo
        self.__normal = normal
        self.__roughness = roughness
        self.__metalicity = metalicity
        self.__emissivity = emissivity
        self.__occlusion = occlusion

    @staticmethod
    def get_fragment() -> str:
        return """
            struct TexId {
                albedo: i32,
                normal: i32,
                roughness: i32,
                metalicity: i32,
                emissivity: i32,
                occlusion: i32,
            }

            @group(1) @binding(0) var textures: texture_2d_array<f32>;
            @group(1) @binding(1) var samplr: sampler;
            @group(1) @binding(2) var<storage, read> tex_ids: array<TexId>;

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
                let F0 = vec3<f32>(0.15); // Base Reflectance
                let lightColor = lights[0].color.rgb * PI;
                var id = i32(in.mat_id);
                let emissivity = textureSample(textures, samplr, in.uv, tex_ids[id].emissivity).rgb;
                let metalicity = textureSample(textures, samplr, in.uv, tex_ids[id].metalicity).r;
                var albedo = textureSample(textures, samplr, in.uv, tex_ids[id].albedo).rgb;
                albedo = pow(albedo, vec3<f32>(2.2)); // gamma correction

                let roughness = textureSample(textures, samplr, in.uv, tex_ids[id].roughness).r;

                let alpha = roughness * roughness;

                // normal map
                let normal_from_map = 2.0 * textureSample(textures, samplr, in.uv, tex_ids[id].normal).rgb - 1.0;
                let N = normalize(normal_from_map.x * in.tangent + normal_from_map.y * in.bitangent + normal_from_map.z * in.normal);
                //let N = normalize(in.normal);

                var L = normalize(-lights[0].position.xyz);  // direction light
                if(lights[0].position.w != 0.0) {
                    L = normalize(lights[0].position.xyz - in.position.xyz); // point light
                }

                let V = normalize(camera.position.xyz - in.position.xyz);
                let H = normalize(V + L);

                let occlusion = textureSample(textures, samplr, in.uv, tex_ids[id].occlusion).rgb;

                let ambiantLightColor = vec3<f32>(0.40);
                let ambiant = albedo * ambiantLightColor * occlusion;

                return vec4<f32>(PBR(alpha, F0, N, L, V, H, albedo, lightColor, emissivity, metalicity) + ambiant, 1.0);
            }
        """

    def get_data(self) -> NDArray:
        return np.array([])

    def get_textures(self) -> list[str]:
        return [
            self.__albedo,
            self.__normal,
            self.__roughness,
            self.__metalicity,
            self.__emissivity,
            self.__occlusion,
        ]

    @staticmethod
    def get_data_size() -> int:
        return 0

    @staticmethod
    def get_texture_count() -> int:
        return 6

    @staticmethod
    def get_texture_size() -> tuple[int, int]:
        return (1024, 1024)

    def __eq__(self, other):
        if isinstance(other, PbrMaterial):
            return self.__albedo == other.__albedo
        return False

    def __hash__(self):
        return hash(self.__albedo)

    @staticmethod
    def set_bindings(
        pipeline: AutoRenderPipeline,
        material_buffer: GPUBuffer,
        texture_ids: GPUBuffer,
        texture_array: GPUTexture,
    ):
        pipeline.set_binding_texture(1, 0, texture_array)
        pipeline.set_binding_sampler(1, 1)
        pipeline.set_binding_buffer(1, 2, texture_ids)
