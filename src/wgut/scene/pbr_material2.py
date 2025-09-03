from numpy.typing import NDArray
import numpy as np
from wgpu import GPUBindGroupLayout, GPURenderPassEncoder, GPURenderPipeline, GPUTexture
from renderer2 import SHADER_HEADER, Renderer
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

    
    fn FRAGMENT(
        N: vec3<f32>, T: vec3<f32>, B: vec3<f32>, V: vec3<f32>, light: Light, color: vec3<f32>, uv: vec2<f32>, mat_id: i32
    ) -> vec3<f32> {
        let albedo = sample(data[mat_id].albedo_tex, uv) * data[mat_id].albedo_color;
        let emissivity = sample(data[mat_id].emissivity_tex, uv) * data[mat_id].emissivity_color;
        let normal = sample(data[mat_id].normal_tex, uv) * data[mat_id].normal_value;
        let roughness = sample(data[mat_id].roughness_tex, uv).x * data[mat_id].roughness_value;
        let metalicity = sample(data[mat_id].metalicity_tex, uv).x * data[mat_id].metalicity_value;
        let occlusion = sample(data[mat_id].occlusion_tex, uv);

        return albedo * occlusion;
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
