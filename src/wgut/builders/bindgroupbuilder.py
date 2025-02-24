from __future__ import annotations
import wgpu

from .builderbase import BuilderBase
from .samplerbuilder import SamplerBuilder
from ..core import get_device

from typing import Self


class BindGroupBuilder(BuilderBase):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.bindings = []
        self.index = 0

    def with_buffer(
        self, buffer: wgpu.GPUBuffer, size=None, offset=0, index=None
    ) -> Self:
        if size is None:
            size = buffer.size - offset
        else:
            size = min(size, buffer.size - offset)

        if index is None:
            index = self.index

        self.bindings.append(
            {
                "binding": index,
                "resource": {
                    "buffer": buffer,
                    "offset": offset,
                    "size": size,
                },
            }
        )

        self.index = index + 1

        return self

    def with_texture(self, texture: wgpu.GPUTexture, index=None) -> Self:
        if index is None:
            index = self.index
        self.bindings.append({"binding": index, "resource": texture.create_view()})
        self.index = index + 1
        return self

    def with_sampler(
        self,
        sampler: wgpu.GPUSampler | None = None,
        index=None,
    ) -> Self:
        if index is None:
            index = self.index
        if sampler is None:
            sampler = SamplerBuilder().build()
        self.bindings.append(
            {
                "binding": index,
                "resource": sampler,
            }
        )
        self.index = index + 1
        return self

    def build(self) -> wgpu.GPUBindGroup:
        return get_device().create_bind_group(
            label=self.label, layout=self.layout, entries=self.bindings
        )
