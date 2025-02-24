from __future__ import annotations
import wgpu

from .builderbase import BuilderBase
from ..core import get_device

from typing import Self


class PipelineLayoutBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.layouts = []

    def with_bind_group_layout(
        self, layouts: wgpu.GPUBindGroupLayout | list[wgpu.GPUBindGroupLayout]
    ) -> Self:
        if not isinstance(layouts, list):
            layouts = [layouts]
        self.layouts += layouts
        return self

    def build(self) -> wgpu.GPUPipelineLayout:
        return get_device().create_pipeline_layout(
            label=self.label, bind_group_layouts=self.layouts
        )
