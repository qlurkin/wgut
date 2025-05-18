from typing import Type
from wgpu import BufferUsage, GPUTexture, TextureFormat, TextureUsage
import numpy.typing as npt
import numpy as np
import random

from wgut.builders.texturebuilder import TextureBuilder
from wgut.core import load_image, write_buffer, write_texture
from wgut.scene.material import Material
from wgut.scene.transform import Transform
from wgut.scene.mesh import Mesh
from wgut.scene.mesh import get_vertex_buffer_descriptor
from wgut.auto_render_pipeline import AutoRenderPipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.builders.vertexbufferdescriptorsbuilder import VertexBufferDescriptorsBuilder
from wgut.camera import Camera


SHADER_START = """
struct Camera {
    matrix: mat4x4<f32>,
    position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

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
    out.pos = camera.matrix * in.position;
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
        vertex_buffer_size: int,
        index_buffer_size: int,
        material_buffer_size: int,
        texture_array_size: tuple[int, int, int],
    ):
        self.__texture_array_size = texture_array_size
        self.__vertex_buffer_size = vertex_buffer_size
        self.__index_buffer_size = index_buffer_size
        self.__material_buffer_size = material_buffer_size

        vertex_buffer_descriptor = get_vertex_buffer_descriptor()
        self.__vertex_buffer_descriptors = (
            VertexBufferDescriptorsBuilder()
            .with_vertex_descriptor(vertex_buffer_descriptor)
            .build()
        )
        self.__vertex_size = vertex_buffer_descriptor["array_stride"]
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
        self.__material_buffer = (
            BufferBuilder()
            .with_size(material_buffer_size)
            .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_DST)
            .build()
        )
        self.__textures = (
            TextureBuilder()
            .with_size(texture_array_size)
            .with_format(TextureFormat.rgba8unorm)
            .with_usage(TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING)
            .build()
        )
        self.__texture_ids_buffer = (
            BufferBuilder()
            .with_size(texture_array_size[2] * 4)
            .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_DST)
            .build()
        )

        self.__vertex_count = 0
        self.__index_count = 0
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0
        self.__frame_stat = None
        self.__clear = True
        self.__material_index = {}
        self.__pipelines = {}
        self.__meshes = {}
        self.__depth_texture = None
        self.__texture_atlas = {}
        self.__texture_ids = []
        self.__texture_names: list[str | None] = [None] * texture_array_size[2]

    def create_pipeline(self, material_class: Type[Material]):
        if material_class not in self.__pipelines:
            shader_source = SHADER_START + material_class.get_fragment()
            pipeline = AutoRenderPipeline(shader_source).with_depth_texture()
            pipeline.set_vertex_buffer_descriptors(self.__vertex_buffer_descriptors)
            pipeline.set_vertex_buffer(0, self.__vertex_buffer)
            pipeline.set_index_buffer(self.__index_buffer)
            material_class.set_bindings(
                pipeline,
                self.__material_buffer,
                self.__texture_ids_buffer,
                self.__textures,
            )
            self.__pipelines[material_class] = pipeline

    def begin_frame(self):
        self.__meshes = {}

    def add_mesh(self, mesh: Mesh, transform: Transform, material: Material):
        cls = type(material)
        if cls not in self.__meshes:
            if material.get_data_size() > self.__material_buffer_size:
                raise IndexError("Material Buffer Not Long Enough")
            if material.get_texture_count() > self.__texture_array_size[2]:
                raise IndexError("Texture Array Not Long Enough")
            self.create_pipeline(cls)
            self.__meshes[cls] = []
        self.__meshes[cls].append((mesh, transform, material))

    def __add_mesh(
        self,
        pipeline: AutoRenderPipeline,
        output_texture: GPUTexture,
        mesh: Mesh,
        transform: Transform,
        material: Material,
    ):
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
                and (len(self.__material_index) + 1) * material.get_data_size()
                > self.__material_buffer_size
            )
            or (
                material not in self.__material_index
                and (len(self.__material_index) + 1) * material.get_texture_count()
                > self.__texture_array_size[2]
            )
        ):
            self.__draw(pipeline, output_texture)

        if material not in self.__material_index:
            material_index = len(self.__material_index)

            if self.__material_buffer is not None:
                if material.get_data_size() > 0:
                    write_buffer(
                        self.__material_buffer,
                        material.get_data(),
                        material_index * material.get_data_size(),
                    )
            self.__material_index[material] = material_index

            if self.__textures is not None:
                images = material.get_textures()
                for img in images:
                    if img not in self.__texture_atlas:
                        ids = set(range(self.__texture_array_size[2]))
                        ids = ids - set(self.__texture_ids)
                        id = random.choice(list(ids))
                        old_name = self.__texture_names[id]
                        if old_name is not None:
                            del self.__texture_atlas[old_name]
                        self.__texture_names[id] = img
                        self.__texture_atlas[img] = id
                        write_texture(self.__textures, load_image(img), id)
                    self.__texture_ids.append(self.__texture_atlas[img])
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

    def end_frame(self, output_texture: GPUTexture, camera: Camera):
        self.__clear = True
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0

        if (
            self.__depth_texture is None
            or self.__depth_texture.size != output_texture.size
        ):
            self.__depth_texture = TextureBuilder().build_depth(output_texture.size)

        view_matrix, proj_matrix = camera.get_matrices(
            output_texture.width / output_texture.height
        )
        camera_position = np.hstack([camera.get_position(), [1.0]]).astype(np.float32)
        camera_matrix = np.array(proj_matrix @ view_matrix, dtype=np.float32)

        # Must send transpose version of matrices, because GPU expect matrices in column major order
        camera_data = np.vstack(
            [np.ascontiguousarray(camera_matrix.T), camera_position]
        )

        for material_class in self.__meshes:
            pipeline = self.__pipelines[material_class]
            pipeline.set_depth_texture(self.__depth_texture)
            pipeline.set_binding_array(0, 0, camera_data)
            for mesh, transform, material in self.__meshes[material_class]:
                self.__add_mesh(pipeline, output_texture, mesh, transform, material)
            self.__draw(pipeline, output_texture)

        self.__frame_stat = {
            "draw": self.__frame_draw_count,
            "mesh": self.__frame_mesh_count,
            "triangle": self.__frame_triangle_count,
            "vertex": self.__frame_vertex_count,
        }

    def __draw(self, pipeline: AutoRenderPipeline, output_texture: GPUTexture):
        if len(self.__texture_ids) > 0:
            write_buffer(
                self.__texture_ids_buffer, np.array(self.__texture_ids, dtype=np.int32)
            )

        pipeline.render(output_texture, self.__index_count, clear=self.__clear)

        self.__clear = False
        self.__frame_draw_count += 1
        self.__vertex_count = 0
        self.__index_count = 0
        self.__material_index = {}
        self.__texture_ids = []

    def get_frame_stat(self):
        return self.__frame_stat
