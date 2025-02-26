from wgpu import BufferUsage, GPUTexture

from wgut.scene.mesh import Mesh
from wgut.scene.mesh import get_vertex_buffer_descriptor
from wgut.auto_render_pipeline import AutoRenderPipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.builders.vertexbufferdescriptorsbuilder import VertexBufferDescriptorsBuilder


class Renderer:
    def __init__(
        self, shader_source: str, vertex_buffer_size: int, index_buffer_size: int
    ):
        self.__pipeline = AutoRenderPipeline(shader_source)
        vertex_buffer_descriptor = get_vertex_buffer_descriptor()
        vertex_buffer_descriptors = (
            VertexBufferDescriptorsBuilder()
            .with_vertex_descriptor(vertex_buffer_descriptor)
            .build()
        )
        vertex_size = vertex_buffer_descriptor["array_stride"]
        self.__pipeline.set_vertex_buffer_descriptors(vertex_buffer_descriptors)
        self.__vertex_buffer = (
            BufferBuilder()
            .with_size(vertex_buffer_size * vertex_size)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__index_buffer = (
            BufferBuilder()
            .with_size(index_buffer_size)
            .with_usage(BufferUsage.INDEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_count = 0
        self.__pipeline.set_vertex_buffer(0, self.__vertex_buffer)
        self.__pipeline.set_index_buffer(self.__index_buffer)
        self.__output_texture = None

    def begin_frame(self, texture: GPUTexture):
        self.__output_texture = texture

    def add_mesh(self, mesh: Mesh):
        if self.__output_texture is None:
            raise Exception("You must call begin_frame before adding meshes")

    def end_frame(self):
        self.__draw()
        self.__vertex_count = 0
        self.__output_texture = None

    def __draw(self):
        assert self.__output_texture is not None

        self.__pipeline.render(self.__output_texture, self.__vertex_count)
