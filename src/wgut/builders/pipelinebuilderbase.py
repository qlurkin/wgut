from __future__ import annotations
import wgpu

from .builderbase import BuilderBase
from ..core import get_device
from ..slang import compile_slang

from typing import Self


class PipelineBuilderBase(BuilderBase):
    def __init__(self):
        super().__init__()
        self.shader_module = None
        self.shader_source = ""
        self.layout = "auto"

    def with_layout(self, layout: wgpu.GPUPipelineLayout) -> Self:
        self.layout = layout
        return self

    def with_shader_source(self, source: str) -> Self:
        self.shader_source = source
        self.shader_module = get_device().create_shader_module(code=source)
        return self

    def with_slangc(self, filename) -> Self:
        source = compile_slang(filename)
        self.with_shader_source(source)
        return self
