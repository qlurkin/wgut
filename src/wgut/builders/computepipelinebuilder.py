from __future__ import annotations
import wgpu

from .pipelinebuilderbase import PipelineBuilderBase
from ..core import get_device


class ComputePipelineBuilder(PipelineBuilderBase):
    def __init__(self):
        super().__init__()

    def build(self) -> wgpu.GPURenderPipeline:
        return get_device().create_compute_pipeline(
            label=self.label,
            layout=self.layout,  # type: ignore
            compute={"module": self.shader_module, "entry_point": "main"},
        )
