from wgpu import BufferUsage, GPUTexture
import numpy.typing as npt
import numpy as np

from wgut.core import write_buffer
from wgut.scene.transform import Transform
from wgut.scene.mesh import Mesh
from wgut.scene.mesh import get_vertex_buffer_descriptor
from wgut.auto_render_pipeline import AutoRenderPipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.builders.vertexbufferdescriptorsbuilder import VertexBufferDescriptorsBuilder


class Renderer:
    def __init__(
        self, shader_source: str, vertex_buffer_size: int, index_buffer_size: int
    ):
        self.__vertex_buffer_size = vertex_buffer_size
        self.__index_buffer_size = index_buffer_size
        self.__pipeline = AutoRenderPipeline(shader_source)
        vertex_buffer_descriptor = get_vertex_buffer_descriptor()
        vertex_buffer_descriptors = (
            VertexBufferDescriptorsBuilder()
            .with_vertex_descriptor(vertex_buffer_descriptor)
            .build()
        )
        self.__vertex_size = vertex_buffer_descriptor["array_stride"]
        self.__pipeline.set_vertex_buffer_descriptors(vertex_buffer_descriptors)
        self.__vertex_buffer = (
            BufferBuilder()
            .with_size(vertex_buffer_size * self.__vertex_size)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__index_buffer = (
            BufferBuilder()
            .with_size(index_buffer_size * 4)
            .with_usage(BufferUsage.INDEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_count = 0
        self.__index_count = 0
        self.__pipeline.set_vertex_buffer(0, self.__vertex_buffer)
        self.__pipeline.set_index_buffer(self.__index_buffer)
        self.__output_texture = None
        self.__output_size = None
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0
        self.__frame_stat = None
        self.__clear = True

    def begin_frame(self, texture: GPUTexture, camera_matrix: npt.NDArray):
        # Must send transpose version of matrices, because GPU expect matrices in column major order
        self.__pipeline.set_binding_array(0, 0, np.ascontiguousarray(camera_matrix.T))
        self.__output_texture = texture
        if self.__output_size is None or self.__output_size != texture.size[:2]:
            self.__output_size = texture.size[:2]
            self.__pipeline.create_depth_texture(self.__output_size)
        self.__clear = True

    def add_mesh(self, mesh: Mesh, transform: Transform):
        if self.__output_texture is None:
            raise Exception("You must call begin_frame before adding meshes")

        vertex_data = mesh.get_transformed_vertices(transform.get_matrix())
        index_data = mesh.get_indices()

        if len(vertex_data) > self.__vertex_buffer_size:
            raise IndexError("Vertex Buffer Not Long Enough")

        if len(index_data) > self.__index_buffer_size:
            raise IndexError("index Buffer Not Long Enough")

        if (
            self.__vertex_count + len(vertex_data) > self.__vertex_buffer_size
            or self.__index_count + len(index_data) > self.__index_buffer_size
        ):
            self.__draw()

        index_data = index_data + self.__vertex_count

        buffer_offset = self.__vertex_count * self.__vertex_size
        write_buffer(self.__vertex_buffer, vertex_data, buffer_offset)

        buffer_offset = self.__index_count * 4
        write_buffer(self.__index_buffer, index_data, buffer_offset)

        self.__vertex_count += len(vertex_data)
        self.__index_count += len(index_data)
        self.__frame_mesh_count += 1
        self.__frame_triangle_count += len(index_data) // 3
        self.__frame_vertex_count += len(vertex_data)

    def end_frame(self):
        self.__draw()
        self.__frame_stat = {
            "draw": self.__frame_draw_count,
            "mesh": self.__frame_mesh_count,
            "triangle": self.__frame_triangle_count,
            "vertex": self.__frame_vertex_count,
        }
        self.__output_texture = None
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0

    def __draw(self):
        assert self.__output_texture is not None

        self.__pipeline.render(
            self.__output_texture, self.__index_count, clear=self.__clear
        )
        self.__clear = False
        self.__frame_draw_count += 1
        self.__vertex_count = 0
        self.__index_count = 0

    def set_binding_array(self, group: int, binding: int, array: npt.NDArray):
        self.__pipeline.set_binding_array(group, binding, array)

    def get_frame_stat(self):
        return self.__frame_stat
