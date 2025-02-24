from __future__ import annotations
import wgpu

from .builderbase import BuilderBase
import wgut.builders.commandbufferbuilder as cbb

from typing import Self


class RenderPassBuilder(BuilderBase):
    def __init__(
        self, texture: wgpu.GPUTexture, command_buffer_builder: cbb.CommandBufferBuilder
    ):
        super().__init__()
        self.command_buffer_builder = command_buffer_builder
        self.texture = texture
        self.clear_value = (0.9, 0.9, 0.9, 1)
        self.load_op = wgpu.LoadOp.clear
        self.store_op = wgpu.StoreOp.store
        self.depth_stencil_attachment = None

    def with_clear_value(self, value: tuple) -> Self:
        self.clear_value = value
        return self

    def with_load_op(self, load_op: wgpu.LoadOp | str) -> Self:
        self.load_op = load_op
        return self

    def with_store_op(self, store_op: wgpu.StoreOp | str) -> Self:
        self.store_op = store_op
        return self

    def with_depth_stencil(self, texture: wgpu.GPUTexture) -> Self:
        self.depth_stencil_attachment = {
            "view": texture.create_view(),
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        }
        return self

    def build(self) -> wgpu.GPURenderPassEncoder:
        return self.command_buffer_builder.command_encoder.begin_render_pass(
            label=self.label,
            color_attachments=[
                {
                    "view": self.texture.create_view(),
                    "resolve_target": None,
                    "clear_value": self.clear_value,
                    "load_op": self.load_op,
                    "store_op": self.store_op,
                }
            ],
            depth_stencil_attachment=self.depth_stencil_attachment,
        )
