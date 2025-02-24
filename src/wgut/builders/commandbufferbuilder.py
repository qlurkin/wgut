from __future__ import annotations
import wgpu

from .builderbase import BuilderBase
import wgut.builders.renderpassbuilder as rpb
import wgut.builders.computepassbuilder as cpb
from ..core import get_device

from typing import Self


class CommandBufferBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.command_encoder = get_device().create_command_encoder(
            label=self.label,
        )

    def begin_render_pass(self, texture: wgpu.GPUTexture) -> rpb.RenderPassBuilder:
        return rpb.RenderPassBuilder(texture, self)

    def begin_compute_pass(self) -> cpb.ComputePassBuilder:
        return cpb.ComputePassBuilder(self)

    def build(self) -> Self:
        """NOOP -> Call Submit()"""
        return self

    def submit(self):
        get_device().queue.submit([self.command_encoder.finish()])
