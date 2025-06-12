from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from wgpu import GPUBuffer, GPUTexture
from wgut.auto_render_pipeline import AutoRenderPipeline


class PbrMaterial:
    def __init__(
        self,
        albedo: str | tuple[float, float, float, float],
        normal: str | None,
        roughness: str | float,
        metalicity: str | float,
        emissivity: str | tuple[float, float, float],
        occlusion: str | None,
    ):
        self.__albedo_filename = albedo if isinstance(albedo, str) else ""
        self.__normal_filename = normal if isinstance(normal, str) else ""
        self.__roughness_filename = roughness if isinstance(roughness, str) else ""
        self.__metalicity_filename = metalicity if isinstance(metalicity, str) else ""
        self.__emissivity_filename = emissivity if isinstance(emissivity, str) else ""
        self.__occlusion_filename = occlusion if isinstance(occlusion, str) else ""

        self.__albedo_data = (
            albedo if not isinstance(albedo, str) else (0.0, 0.0, 0.0, 0.0)
        )
        self.__normal_data = (0.5, 0.5, 1.0, 1.0)
        self.__roughness_data = roughness if not isinstance(roughness, str) else 0.0
        self.__metalicity_data = metalicity if not isinstance(metalicity, str) else 0.0
        self.__emissivity_data = (
            emissivity + (1.0,)
            if not isinstance(emissivity, str)
            else (0.0, 0.0, 0.0, 0.0)
        )
        self.__occlusion_data = 1.0

        self.__texture_count = len(
            list(
                filter(
                    lambda v: isinstance(v, str),
                    [albedo, normal, roughness, metalicity, emissivity, occlusion],
                )
            )
        )

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
            };

            struct MatData {
                albedo: vec4<f32>,
                normal: vec4<f32>,
                emissivity: vec4<f32>,
                roughness: f32,
                metalicity: f32,
                occlusion: f32,
                padding: f32,
            };

            @group(1) @binding(0) var textures: texture_2d_array<f32>;
            @group(1) @binding(1) var samplr: sampler;
            @group(1) @binding(2) var<storage, read> tex_ids: array<TexId>;
            @group(1) @binding(3) var<storage, read> mat_data: array<MatData>;

            fn get_albedo(uv: vec2<f32>, mat_id: i32) -> vec3<f32> {
                if (tex_ids[mat_id].albedo < 0) {
                    return mat_data[mat_id].albedo.rgb;
                }
                return textureSample(textures, samplr, uv, tex_ids[mat_id].albedo).rgb;
            }

            fn get_normal(uv: vec2<f32>, mat_id: i32) -> vec3<f32> {
                if (tex_ids[mat_id].normal < 0) {
                    return mat_data[mat_id].normal.rgb;
                }
                return textureSample(textures, samplr, uv, tex_ids[mat_id].normal).rgb;
            }

            fn get_roughness(uv: vec2<f32>, mat_id: i32) -> f32 {
                if (tex_ids[mat_id].roughness < 0) {
                    return mat_data[mat_id].roughness;
                }
                return textureSample(textures, samplr, uv, tex_ids[mat_id].roughness).r;
            }

            fn get_metalicity(uv: vec2<f32>, mat_id: i32) -> f32 {
                if (tex_ids[mat_id].metalicity < 0) {
                    return mat_data[mat_id].metalicity;
                }
                return textureSample(textures, samplr, uv, tex_ids[mat_id].metalicity).r;
            }

            fn get_emissivity(uv: vec2<f32>, mat_id: i32) -> vec3<f32> {
                if (tex_ids[mat_id].emissivity < 0) {
                    return mat_data[mat_id].emissivity.rgb;
                }
                return textureSample(textures, samplr, uv, tex_ids[mat_id].emissivity).rgb;
            }

            fn get_occlusion(uv: vec2<f32>, mat_id: i32) -> f32 {
                if (tex_ids[mat_id].occlusion < 0) {
                    return mat_data[mat_id].occlusion;
                }
                return textureSample(textures, samplr, uv, tex_ids[mat_id].occlusion).r;
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

                var albedo = pow(get_albedo(in.uv, id), vec3<f32>(2.2));  // gamma correction
                let normal = 2.0 * get_normal(in.uv, id) - 1.0;
                let roughness = get_roughness(in.uv, id);
                let metalicity = get_metalicity(in.uv, id);
                let emissivity = get_emissivity(in.uv, id);
                let occlusion = get_occlusion(in.uv, id);

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
                

                // let ambiantLightColor = vec3<f32>(0.40);
                // let ambiant = albedo * ambiantLightColor * occlusion;

                return vec4<f32>(color, 1.0);
            }
        """

    def get_data(self) -> NDArray:
        data = np.hstack(
            [
                self.__albedo_data,
                self.__normal_data,
                self.__emissivity_data,
                self.__roughness_data,
                self.__metalicity_data,
                self.__occlusion_data,
                [0.0],
            ]
        ).astype(np.float32)

        return data

    def get_textures(self) -> list[str]:
        return [
            self.__albedo_filename,
            self.__normal_filename,
            self.__roughness_filename,
            self.__metalicity_filename,
            self.__emissivity_filename,
            self.__occlusion_filename,
        ]

    @staticmethod
    def get_data_size() -> int:
        return 64

    def get_texture_count(self) -> int:
        return self.__texture_count

    @staticmethod
    def get_texture_size() -> tuple[int, int]:
        return (1024, 1024)

    def __as_tuple(self) -> tuple:
        return (
            self.__albedo_filename
            if len(self.__albedo_filename) > 0
            else self.__albedo_data,
            self.__normal_filename
            if len(self.__normal_filename) > 0
            else self.__normal_data,
            self.__roughness_filename
            if len(self.__roughness_filename) > 0
            else self.__roughness_data,
            self.__metalicity_filename
            if len(self.__metalicity_filename) > 0
            else self.__metalicity_data,
            self.__emissivity_filename
            if len(self.__emissivity_filename) > 0
            else self.__emissivity_data,
            self.__occlusion_filename
            if len(self.__occlusion_filename) > 0
            else self.__occlusion_data,
        )

    def __eq__(self, other):
        if isinstance(other, PbrMaterial):
            return self.__as_tuple() == other.__as_tuple()
        return False

    def __hash__(self):
        return hash(self.__as_tuple())

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
        pipeline.set_binding_buffer(1, 3, material_buffer)
