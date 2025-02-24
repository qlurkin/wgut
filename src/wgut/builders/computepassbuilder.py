from __future__ import annotations

from .builderbase import BuilderBase
import wgut.builders.commandbufferbuilder as cbb


class ComputePassBuilder(BuilderBase):
    def __init__(self, command_buffer_builder: cbb.CommandBufferBuilder):
        super().__init__()
        self.command_buffer_builder = command_buffer_builder

    def build(self):
        return self.command_buffer_builder.command_encoder.begin_compute_pass(
            label=self.label,
        )
