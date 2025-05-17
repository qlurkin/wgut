import numpy as np
from numpy.typing import NDArray
from wgpu import GPUBuffer, GPUTexture

from wgut.auto_render_pipeline import AutoRenderPipeline


class BasicColorMaterial:
    def __init__(self, color: tuple[float, float, float]):
        self.__color = color
        self.__data = np.array(color + (1.0,), dtype=np.float32)

    @staticmethod
    def get_fragment() -> str:
        return """
            struct Material {
                color: vec4<f32>,
            }

            @group(1) @binding(0) var<storage, read> materials: array<Material>;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                var color = materials[u32(in.mat_id)].color;
                color = vec4<f32>(pow(color.rgb, vec3<f32>(2.2)), 1.0);
                return color;
            }
        """

    def get_data(self) -> NDArray:
        return self.__data

    def get_textures(self) -> list[str]:
        return []

    @staticmethod
    def get_data_size() -> int:
        return 16

    @staticmethod
    def get_texture_count() -> int:
        return 0

    @staticmethod
    def get_texture_size() -> tuple[int, int]:
        return (0, 0)

    def __eq__(self, other):
        return self.__color == other.__color

    def __hash__(self):
        return hash(self.__color)

    @staticmethod
    def set_bindings(
        pipeline: AutoRenderPipeline,
        material_buffer: GPUBuffer,
        texture_ids: GPUBuffer,
        texture_array: GPUTexture,
    ):
        pipeline.set_binding_buffer(1, 0, material_buffer)
