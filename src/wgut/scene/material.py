from typing import Protocol
import numpy.typing as npt
from wgpu import GPUBuffer, GPUTexture

from wgut.auto_render_pipeline import AutoRenderPipeline


class Material(Protocol):
    @staticmethod
    def get_fragment() -> str: ...
    @staticmethod
    def get_data_size() -> int: ...
    def get_texture_count(self) -> int: ...
    @staticmethod
    def get_texture_size() -> tuple[int, int]: ...
    def get_data(self) -> npt.NDArray: ...
    def get_textures(self) -> list[str]: ...
    @staticmethod
    def set_bindings(
        pipeline: AutoRenderPipeline,
        material_buffer: GPUBuffer,
        texture_ids: GPUBuffer,
        texture_array: GPUTexture,
    ): ...
