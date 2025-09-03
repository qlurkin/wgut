from typing import Protocol

from wgpu import (
    GPUBindGroupLayout,
    GPURenderPassEncoder,
    GPUTexture,
)


class Material(Protocol):
    @staticmethod
    def get_fragment() -> str: ...

    @staticmethod
    def get_bind_group_layouts() -> list[GPUBindGroupLayout]: ...

    @staticmethod
    def get_data_size() -> int: ...

    def get_texture_count(self) -> int: ...

    def get_data(self, tex_ids: list[int]) -> bytes: ...

    def get_textures(self) -> list[GPUTexture]: ...

    @staticmethod
    def set_bindings(render_pass: GPURenderPassEncoder, bind_group_offset=0): ...
