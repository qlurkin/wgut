from __future__ import annotations
from PIL.Image import Image
import numpy as np
from numpy.typing import NDArray
from wgpu import GPUBuffer, GPUTexture
from wgut.auto_render_pipeline import AutoRenderPipeline
from wgut.core import load_image


class PbrMaterial:
    def __init__(self, filename: str):
        self.__image = load_image(filename)
        self.__filename = filename

    @staticmethod
    def get_fragment() -> str:
        return """
            @group(1) @binding(0) var textures: texture_2d_array<f32>;
            @group(1) @binding(1) var samplr: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                var id = i32(in.mat_id);
                var color = textureSample(textures, samplr, in.uv, id);
                color = vec4<f32>(pow(color.rgb, vec3<f32>(2.2)), 1.0);
                return color;
            }
        """

    def get_data(self) -> NDArray:
        return np.array([])

    def get_textures(self) -> list[Image]:
        return [self.__image]

    @staticmethod
    def get_data_size() -> int:
        return 0

    @staticmethod
    def get_texture_count() -> int:
        return 1

    @staticmethod
    def get_texture_size() -> tuple[int, int]:
        return (1024, 1024)

    def get_filename(self) -> str:
        return self.__filename

    def __eq__(self, other):
        if isinstance(other, PbrMaterial):
            return self.__filename == other.get_filename()
        return False

    def __hash__(self):
        return hash(self.__filename)

    @staticmethod
    def set_bindings(
        pipeline: AutoRenderPipeline,
        material_buffer: GPUBuffer,
        texture_array: GPUTexture,
    ):
        pipeline.set_binding_texture(1, 0, texture_array)
        pipeline.set_binding_sampler(1, 1)
