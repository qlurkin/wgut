from __future__ import annotations
import wgpu

from .builderbase import BuilderBase
from ..core import get_device

from typing import Self


class BingGroupLayoutBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.entries = []
        self.index = 0

    def with_buffer(
        self,
        visibility: wgpu.ShaderStage | int,
        type: wgpu.BufferBindingType | str = "uniform",
        index: int | None = None,
    ) -> Self:
        if index is None:
            index = self.index
        self.entries.append(
            {
                "binding": index,
                "visibility": visibility,
                "buffer": {"type": type},
            }
        )
        self.index = index + 1
        return self

    def with_sampler(
        self,
        visibility: wgpu.ShaderStage | int,
        index: int | None = None,
    ) -> Self:
        if index is None:
            index = self.index
        self.entries.append(
            {
                "binding": index,
                "visibility": visibility,
                "sampler": {},
            }
        )
        self.index = index + 1
        return self

    def with_texture_2d(
        self,
        visibility: wgpu.ShaderStage | int,
        index: int | None = None,
    ) -> Self:
        if index is None:
            index = self.index
        self.entries.append(
            {
                "binding": index,
                "visibility": visibility,
                "texture": {},
            }
        )
        self.index = index + 1
        return self

    def with_texture_2d_array(
        self,
        visibility: wgpu.ShaderStage | int,
        index: int | None = None,
    ) -> Self:
        if index is None:
            index = self.index
        self.entries.append(
            {
                "binding": index,
                "visibility": visibility,
                "texture": {"view_dimension": wgpu.TextureViewDimension.d2_array},
            }
        )
        self.index = index + 1
        return self

    def build(self) -> wgpu.GPUBindGroupLayout:
        return get_device().create_bind_group_layout(
            label=self.label, entries=self.entries
        )
