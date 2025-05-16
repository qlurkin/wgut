from typing import Type
from wgpu import BufferUsage, GPUTexture, TextureFormat, TextureUsage, VertexFormat
import numpy.typing as npt
import numpy as np

from wgut.builders.texturebuilder import TextureBuilder
from wgut.core import write_buffer, write_texture
from wgut.scene.material import Material
from wgut.scene.transform import Transform
from wgut.scene.mesh import Mesh
from wgut.scene.mesh import get_vertex_buffer_descriptor
from wgut.auto_render_pipeline import AutoRenderPipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.builders.vertexbufferdescriptorsbuilder import VertexBufferDescriptorsBuilder


SHADER_START = """
@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) bitangent: vec3<f32>,
    @location(6) mat_id: f32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) mat_id: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = camera * in.position;
    out.color = in.color;
    out.uv = in.uv;
    out.normal = in.normal;
    out.tangent = in.tangent;
    out.bitangent = in.bitangent;
    out.mat_id = in.mat_id;
    return out;
}


"""


class Renderer:
    def __init__(
        self,
        material_class: Type[Material],
        vertex_buffer_size: int,
        index_buffer_size: int,
        material_buffer_size: int,
    ):
        shader_source = SHADER_START + material_class.get_fragment()
        self.__material_size = material_class.get_data_size()
        self.__material_texture_count = material_class.get_texture_count()
        self.__texture_array_size = material_class.get_texture_size() + (
            self.__material_texture_count * material_buffer_size,
        )
        self.__vertex_buffer_size = vertex_buffer_size
        self.__index_buffer_size = index_buffer_size
        self.__material_buffer_size = material_buffer_size

        self.__pipeline = AutoRenderPipeline(shader_source).with_depth_texture()
        vertex_buffer_descriptor = get_vertex_buffer_descriptor()
        vertex_buffer_descriptor["attributes"].append(  # Add material_index to vertex
            {
                "format": VertexFormat.float32,
                "offset": (4 + 4 + 2 + 3 + 3 + 3) * 4,
                "shader_location": 6,
            },
        )
        vertex_buffer_descriptor["array_stride"] += 4
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

        self.__material_buffer = None
        if self.__material_size > 0:
            self.__material_buffer = (
                BufferBuilder()
                .with_size(material_buffer_size * self.__material_size)
                .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_DST)
                .build()
            )

        self.__textures = None
        if self.__material_texture_count > 0:
            self.__textures = (
                TextureBuilder()
                .with_size(self.__texture_array_size)
                .with_format(TextureFormat.rgba8unorm_srgb)
                .with_usage(TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING)
                .build()
            )

        self.__vertex_count = 0
        self.__index_count = 0
        self.__texture_count = 0
        self.__pipeline.set_vertex_buffer(0, self.__vertex_buffer)
        self.__pipeline.set_index_buffer(self.__index_buffer)

        binding_index = 0

        if self.__material_buffer is not None:
            self.__pipeline.set_binding_buffer(1, binding_index, self.__material_buffer)
            binding_index += 1

        if self.__textures is not None:
            self.__pipeline.set_binding_texture(1, binding_index, self.__textures)
            binding_index += 1
            self.__pipeline.set_binding_sampler(1, binding_index)
            binding_index += 1

        self.__output_texture = None
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0
        self.__frame_stat = None
        self.__clear = True
        self.__material_index = {}

    def begin_frame(self, texture: GPUTexture, camera_matrix: npt.NDArray):
        # Must send transpose version of matrices, because GPU expect matrices in column major order
        self.__pipeline.set_binding_array(0, 0, np.ascontiguousarray(camera_matrix.T))
        self.__output_texture = texture
        self.__clear = True
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0

    def add_mesh(self, mesh: Mesh, transform: Transform, material: Material):
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
            or (
                material not in self.__material_index
                and len(self.__material_index) == self.__material_buffer_size
            )
        ):
            self.__draw()

        if material not in self.__material_index:
            material_index = len(self.__material_index)

            if self.__material_buffer is not None:
                write_buffer(
                    self.__material_buffer,
                    material.get_data(),
                    material_index * self.__material_size,
                )
            self.__material_index[material] = material_index

            if self.__textures is not None:
                images = material.get_textures()
                for img in images:
                    write_texture(self.__textures, img, self.__texture_count)
                    self.__texture_count += 1
        else:
            material_index = self.__material_index[material]

        vertex_data = np.hstack(
            [
                vertex_data,
                np.full((len(vertex_data), 1), material_index, dtype=np.float32),
            ]
        )

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

    def __draw(self):
        assert self.__output_texture is not None

        # TODO: create and send material buffer

        self.__pipeline.render(
            self.__output_texture, self.__index_count, clear=self.__clear
        )
        self.__clear = False
        self.__frame_draw_count += 1
        self.__vertex_count = 0
        self.__index_count = 0
        self.__texture_count = 0
        self.__material_index = {}

    def set_binding_array(self, group: int, binding: int, array: npt.NDArray):
        self.__pipeline.set_binding_array(group, binding, array)

    def get_frame_stat(self):
        return self.__frame_stat
