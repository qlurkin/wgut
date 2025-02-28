from __future__ import annotations
import wgpu

from .pipelinebuilderbase import PipelineBuilderBase
from ..core import get_device

from typing import Self

from wgpu.enums import TextureFormat


class RenderPipelineBuilder(PipelineBuilderBase):
    def __init__(self, output_format: TextureFormat | str):
        super().__init__()
        self.buffers = []
        self.location = 0
        self.output_format = output_format
        self.depth_stencil_state = None

    def with_vertex_buffer_descriptors(self, vertex_buffer_descriptors: list) -> Self:
        self.buffers = vertex_buffer_descriptors
        return self

    def with_depth_stencil(
        self,
        depth_format: wgpu.TextureFormat | str = wgpu.TextureFormat.depth32float,
        depth_write_enabled: bool = True,
        depth_compare: wgpu.CompareFunction | str = wgpu.CompareFunction.less,
    ) -> Self:
        self.depth_stencil_state = {
            "format": depth_format,
            "depth_write_enabled": depth_write_enabled,
            "depth_compare": depth_compare,
        }
        return self

    def build(self) -> wgpu.GPURenderPipeline:
        return get_device().create_render_pipeline(
            label=self.label,
            layout=self.layout,  # type: ignore
            vertex={
                "module": self.shader_module,
                "entry_point": "vs_main",
                "buffers": self.buffers,
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil=self.depth_stencil_state,
            multisample=None,
            fragment={
                "module": self.shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.output_format,
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                ],
            },
        )
