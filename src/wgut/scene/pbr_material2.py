from numpy.typing import NDArray
import numpy as np
from wgpu import (
    GPUBindGroupLayout,
    GPURenderPassEncoder,
    GPUTexture,
    TextureFormat,
    TextureUsage,
)

from wgut.core import get_device, load_image, write_texture

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
    padding1: f32,  // should not be mandatory but needed on osX
    padding2: f32,
    padding3: f32,
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
    let mat_id = i32(in.mat_id);

    let albedo = sample(mat_data[mat_id].albedo_tex, in.uv).rgb * mat_data[mat_id].albedo_color;
    let emissivity = sample(mat_data[mat_id].emissivity_tex, in.uv).rgb * mat_data[mat_id].emissivity_color;
    let normal = 2.0 * sample(mat_data[mat_id].normal_tex, in.uv).rgb * mat_data[mat_id].normal_value - 1.0;
    let roughness = sample(mat_data[mat_id].roughness_tex, in.uv).x * mat_data[mat_id].roughness_value;
    let metalicity = sample(mat_data[mat_id].metalicity_tex, in.uv).x * mat_data[mat_id].metalicity_value;
    let occlusion = sample(mat_data[mat_id].occlusion_tex, in.uv).x;

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

    //return textureSample(texture6, samplr, in.uv);
    //return sample(mat_data[mat_id].albedo_tex, in.uv);
    //return vec4<f32>(vec3<f32>(f32(mat_data[mat_id].albedo_tex)/1000000000.0), 1.0);
    return vec4<f32>(color, 1.0);
}
"""


class PbrMaterial:
    def __init__(
        self,
        albedo: str | tuple[float, float, float],
        normal: str | None | tuple[float, float, float],
        roughness: str | float,
        metalicity: str | float,
        emissivity: str | tuple[float, float, float],
        occlusion: str | None | float,
    ):
        if normal is None:
            normal = (0.5, 0.5, 1.0)
        if occlusion is None:
            occlusion = 1.0

        self.__texture_count = 0
        self.__has_none_texture = False

        self.__albedo_texture, self.__albedo_value = self.create_texture(
            albedo, TextureFormat.rgba8unorm_srgb, [1, 1, 1]
        )
        self.__normal_texture, self.__normal_value = self.create_texture(
            normal, TextureFormat.rgba8unorm, [1, 1, 1]
        )
        self.__roughness_texture, self.__roughness_value = self.create_texture(
            roughness, TextureFormat.rgba8unorm, 1
        )
        self.__metalicity_texture, self.__metalicity_value = self.create_texture(
            metalicity, TextureFormat.rgba8unorm, 1
        )
        self.__emissivity_texture, self.__emissivity_value = self.create_texture(
            emissivity, TextureFormat.rgba8unorm, [1, 1, 1]
        )
        self.__occlusion_texture, self.__occlusion_value = self.create_texture(
            occlusion, TextureFormat.rgba8unorm, 1
        )

        if self.__has_none_texture:
            self.__texture_count += 1

    def create_texture(
        self, texture, format: TextureFormat | str, value
    ) -> tuple[GPUTexture | None, NDArray]:
        if isinstance(texture, str):
            self.__texture_count += 1
            img = load_image(texture)
            texture = get_device().create_texture(
                label="Albedo Texture",
                size=img.size + (1,),
                format=format,  # type: ignore
                usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,  # type: ignore
            )
            write_texture(texture, img)
            return texture, np.array(value, dtype=np.float32)
        else:
            self.__has_none_texture = True
            return None, np.array(texture, dtype=np.float32)

    @staticmethod
    def get_fragment() -> str:
        return SHADER

    @staticmethod
    def get_bind_group_layouts() -> list[GPUBindGroupLayout]:
        # To add bind groups
        return []

    @staticmethod
    def get_data_size() -> int:
        return 17 * 4 + 12

    def get_texture_count(self) -> int:
        return self.__texture_count

    def get_data(self, tex_ids: list[int]) -> bytes:
        # must send the textures to the renderer to get the texture's ids and put them in the data
        # struct DATA {
        #     albedo_color: vec3<f32>,
        #     albedo_tex: i32,
        #     emissivity_color: vec3<f32>,
        #     emissivity_tex: i32,
        #     normal_value: vec3<f32>,
        #     normal_tex: i32,
        #     roughness_value: f32,
        #     roughness_tex: i32,
        #     metalicity_value: f32,
        #     metalicity_tex: i32,
        #     occlusion_tex: i32,
        #     // implicit padding of 12 bytes
        # }

        res = b""
        res += self.__albedo_value.tobytes()
        res += np.int32(tex_ids[0]).tobytes()
        res += self.__emissivity_value.tobytes()
        res += np.int32(tex_ids[4]).tobytes()
        res += self.__normal_value.tobytes()
        res += np.int32(tex_ids[1]).tobytes()
        res += self.__roughness_value.tobytes()
        res += np.int32(tex_ids[2]).tobytes()
        res += self.__metalicity_value.tobytes()
        res += np.int32(tex_ids[3]).tobytes()
        res += np.int32(tex_ids[5]).tobytes()
        # res += np.array([0, 0, 0], dtype=np.float32).tobytes()

        return res

    def get_textures(self) -> list[GPUTexture | None]:
        return [
            self.__albedo_texture,
            self.__normal_texture,
            self.__roughness_texture,
            self.__metalicity_texture,
            self.__emissivity_texture,
            self.__occlusion_texture,
        ]

    @staticmethod
    def set_bindings(render_pass: GPURenderPassEncoder, bind_group_offset=0):
        # probably noting to do here but can be usefull if additional bindings is used
        pass
