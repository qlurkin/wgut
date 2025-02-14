from wgut.reflection import Reflection
from wgut.builders import (
    BufferBuilder,
    SamplerBuilder,
    BindGroupBuilder,
    CommandBufferBuilder,
    RenderPipelineBuilder,
    TextureBuilder,
    VertexBufferDescriptorsBuilder,
)
import numpy.typing as npt
import numpy as np
from wgpu import (
    BufferUsage,
    GPUBuffer,
    GPUSampler,
    GPUTexture,
    IndexFormat,
    VertexFormat,
)
from typing import Self


class AutoRenderPipeline:
    def __init__(self, shader_source: str):
        self.wgsl_source = shader_source

        self.reflection = Reflection(self.wgsl_source)
        self.bind_groups = {}
        self.bindings = {}
        for gid in self.reflection.get_bind_group_ids():
            self.bind_groups[gid] = None
            self.bindings[gid] = {}
            for bid in self.reflection.get_binding_ids(gid):
                self.bindings[gid][bid] = None
        self.vertex_buffer_descriptors = []
        self.vertex_buffers: list[GPUBuffer | None] = []
        self.index_buffer = None
        self.index_format = IndexFormat.uint32
        self.depth_texture = None
        self.pipeline = None
        self.output_format = None
        self.vertex_descriptors_builder = None

    def set_vertex_buffer_descriptors(self, vertex_buffer_descriptors: list) -> Self:
        if self.vertex_descriptors_builder is not None:
            raise Exception(
                "You already begin to create Vertex Buffer Descriptor with simple methods. You can't do both."
            )
        self.vertex_buffer_descriptors = vertex_buffer_descriptors
        self.vertex_buffers = [None] * len(
            vertex_buffer_descriptors
        )  # Create slots for vertex buffers
        self.pipeline = None  # must recreate the pipeline
        return self

    def add_simple_vertex_descriptor(self, *vertex_formats: VertexFormat | str) -> Self:
        if self.vertex_descriptors_builder is None:
            self.vertex_descriptors_builder = VertexBufferDescriptorsBuilder()
            self.vertex_buffers = []  # reset the buffers
        self.vertex_descriptors_builder.with_vertex_buffer()
        for format in vertex_formats:
            self.vertex_descriptors_builder.with_attribute(format)
        self.vertex_buffers.append(None)  # Add a slot for the vertex buffer
        return self

    def add_simple_instance_descriptor(
        self, *vertex_formats: VertexFormat | str
    ) -> Self:
        if self.vertex_descriptors_builder is None:
            self.vertex_descriptors_builder = VertexBufferDescriptorsBuilder()
            self.vertex_buffers = []  # reset the buffers
        self.vertex_descriptors_builder.with_instance_buffer()
        for format in vertex_formats:
            self.vertex_descriptors_builder.with_attribute(format)
        self.vertex_buffers.append(None)  # Add a slot for the vertex buffer
        return self

    def set_binding_array(
        self,
        group: int,
        binding: int,
        array: npt.NDArray,
        additional_usages: BufferUsage | int = 0,
    ) -> GPUBuffer:
        builder = BufferBuilder().from_data(array)
        if self.reflection.get_binding_space(group, binding) == "uniform":
            builder.with_usage(BufferUsage.UNIFORM | additional_usages)  # type: ignore
        else:
            builder.with_usage(BufferUsage.STORAGE | additional_usages)  # type: ignore

        buffer = builder.build()
        return self.set_binding_buffer(group, binding, buffer)

    def set_binding_buffer(
        self, group: int, binding: int, buffer: GPUBuffer
    ) -> GPUBuffer:
        self.bind_groups[group] = None
        self.bindings[group][binding] = buffer
        return buffer

    def set_binding_texture(
        self, group: int, binding: int, texture: GPUTexture
    ) -> GPUTexture:
        self.bind_groups[group] = None
        self.bindings[group][binding] = texture
        return texture

    def set_binding_sampler(
        self, group: int, binding: int, sampler: GPUSampler | None = None
    ) -> GPUSampler:
        if sampler is None:
            sampler = SamplerBuilder().build()
        self.bind_groups[group] = None
        self.bindings[group][binding] = sampler
        return sampler

    def set_vertex_buffer(self, id: int, buffer: GPUBuffer) -> GPUBuffer:
        self.vertex_buffers[id] = buffer
        return buffer

    def set_vertex_array(
        self,
        id: int,
        array: npt.NDArray,
        additional_usages: BufferUsage | int = 0,
    ) -> GPUBuffer:
        buffer = (
            BufferBuilder()
            .from_data(array)
            .with_usage(BufferUsage.VERTEX | additional_usages)  # type: ignore
            .build()
        )
        return self.set_vertex_buffer(id, buffer)

    def set_index_buffer(
        self, buffer: GPUBuffer, index_format: IndexFormat | str = IndexFormat.uint32
    ) -> GPUBuffer:
        self.index_buffer = buffer
        self.index_format = index_format
        return buffer

    def set_index_array(
        self, array: npt.NDArray, additional_usages: BufferUsage | int = 0
    ) -> GPUBuffer:
        if array.dtype == np.uint32:
            index_format = IndexFormat.uint32
        elif array.dtype == np.uint16:
            index_format = IndexFormat.uint16
        else:
            raise TypeError(f"Invalid Index Array Type: {array.dtype}")

        buffer = (
            BufferBuilder()
            .from_data(array)
            .with_usage(BufferUsage.INDEX | additional_usages)  # type: ignore
            .build()
        )
        return self.set_index_buffer(buffer, index_format)

    def set_depth_texture(self, texture: GPUTexture) -> GPUTexture:
        if self.depth_texture is None:
            self.pipeline = None  # must recreate the pipeline
        self.depth_texture = texture
        return texture

    def create_depth_texture(self, size: tuple[int, int]) -> GPUTexture:
        texture = TextureBuilder().build_depth(size)
        return self.set_depth_texture(texture)

    def render(
        self, output_texture: GPUTexture, vertex_count: int, instance_count: int = 1
    ):
        if self.output_format is None or self.output_format != output_texture.format:
            self.output_format = output_texture.format
            self.pipeline = None  # must recreate the pipeline

        if self.vertex_descriptors_builder is not None:
            self.vertex_buffer_descriptors = self.vertex_descriptors_builder.build()
            self.vertex_descriptors_builder = None
            self.pipeline = None

        if self.pipeline is None:
            pipeline_builder = (
                RenderPipelineBuilder(self.output_format)
                .with_vertex_buffer_descriptors(self.vertex_buffer_descriptors)
                .with_layout(self.reflection.get_pipeline_layout())
                .with_shader_source(self.wgsl_source)
            )

            if self.depth_texture is not None:
                pipeline_builder.with_depth_stencil()

            self.pipeline = pipeline_builder.build()

        for gid in self.bind_groups:
            if self.bind_groups[gid] is None:
                layout = self.reflection.get_bind_group_layout(gid)
                builder = BindGroupBuilder(layout)
                for bid in self.bindings[gid]:
                    if isinstance(self.bindings[gid][bid], GPUTexture):
                        builder.with_texture(self.bindings[gid][bid], index=bid)
                    elif isinstance(self.bindings[gid][bid], GPUSampler):
                        builder.with_sampler(self.bindings[gid][bid], index=bid)
                    else:
                        builder.with_buffer(self.bindings[gid][bid], index=bid)
                self.bind_groups[gid] = builder.build()
        command_encoder = CommandBufferBuilder()
        render_pass_builder = command_encoder.begin_render_pass(output_texture)
        if self.depth_texture is not None:
            render_pass_builder.with_depth_stencil(self.depth_texture)
        render_pass = render_pass_builder.build()
        render_pass.set_pipeline(self.pipeline)
        for gid, g in self.bind_groups.items():
            render_pass.set_bind_group(gid, g)
        for id, buf in enumerate(self.vertex_buffers):
            if buf is None:
                raise ValueError(f"Vertex Buffer {id} not set")
            render_pass.set_vertex_buffer(id, buf)
        if self.index_buffer is None:
            render_pass.draw(vertex_count, instance_count)
        else:
            render_pass.set_index_buffer(self.index_buffer, self.index_format)  # type: ignore
            render_pass.draw_indexed(vertex_count, instance_count)
        render_pass.end()

        command_encoder.submit()
