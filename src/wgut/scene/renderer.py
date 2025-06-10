import time
from typing import Type
from wgpu import BufferUsage, GPUTexture, LoadOp, TextureFormat, TextureUsage
import random
from pyglm.glm import (
    array,
    float32,
    int32,
    vec4,
)

from wgut.builders.commandbufferbuilder import CommandBufferBuilder
from wgut.builders.texturebuilder import TextureBuilder
from wgut.core import load_image, write_buffer, write_texture
from wgut.scene.material import Material
from wgut.scene.transform import Transform
from wgut.scene.mesh import Mesh, get_vertex_buffer_descriptors
from wgut.auto_render_pipeline import AutoRenderPipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.camera import Camera


SHADER_START = """
struct Camera {
    matrix: mat4x4<f32>,
    position: vec4<f32>,
};

// 4th component of color is intensity, 4th component of position equal to 0 means Direction Light.
// All four component of position to 0 means ambiant light.
struct Light {
    position: vec4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> lights: array<Light>;

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
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) bitangent: vec3<f32>,
    @location(6) mat_id: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = camera.matrix * in.position;
    out.position = in.position;
    out.color = in.color;
    out.uv = in.uv;
    out.normal = in.normal;
    out.tangent = in.tangent;
    out.bitangent = in.bitangent;
    out.mat_id = in.mat_id;
    return out;
}


"""

POSITION_LOCATION = 0
COLOR_LOCATION = 1
UV_LOCATION = 2
NORMAL_LOCATION = 3
TANGENT_LOCATION = 4
BITANGENT_LOCATION = 5
MAT_ID_LOCATION = 6

VERTEX_DATA_SIZE = {
    POSITION_LOCATION: 4 * 4,
    COLOR_LOCATION: 4 * 4,
    UV_LOCATION: 4 * 2,
    NORMAL_LOCATION: 4 * 3,
    TANGENT_LOCATION: 4 * 3,
    BITANGENT_LOCATION: 4 * 3,
    MAT_ID_LOCATION: 4,
}


class Renderer:
    def __init__(
        self,
        vertex_buffer_size: int,
        index_buffer_size: int,
        material_buffer_size: int,
        light_buffer_size: int,
        texture_array_size: tuple[int, int, int],
        texture_ids_buffer_size: int | None = None,
    ):
        if texture_ids_buffer_size is None:
            texture_ids_buffer_size = texture_array_size[2] * 4

        self.__texture_array_size = texture_array_size
        self.__vertex_buffer_size = vertex_buffer_size
        self.__index_buffer_size = index_buffer_size
        self.__material_buffer_size = material_buffer_size

        self.__vertex_buffer_descriptors = get_vertex_buffer_descriptors()

        self.__vertex_buffers = []
        self.__vertex_buffers.append(
            BufferBuilder()
            .with_size(vertex_buffer_size * 4 * 4)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_buffers.append(
            BufferBuilder()
            .with_size(vertex_buffer_size * 4 * 4)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_buffers.append(
            BufferBuilder()
            .with_size(vertex_buffer_size * 4 * 2)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_buffers.append(
            BufferBuilder()
            .with_size(vertex_buffer_size * 4 * 3)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_buffers.append(
            BufferBuilder()
            .with_size(vertex_buffer_size * 4 * 3)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_buffers.append(
            BufferBuilder()
            .with_size(vertex_buffer_size * 4 * 3)
            .with_usage(BufferUsage.VERTEX | BufferUsage.COPY_DST)
            .build()
        )
        self.__vertex_buffers.append(
            BufferBuilder()
            .with_size(vertex_buffer_size * 4)
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
            .with_size(texture_ids_buffer_size)
            .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_DST)
            .build()
        )

        self.__vertex_count = 0
        self.__index_count = 0
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0
        self.__frame_stat = {
            "draw": 0,
            "mesh": 0,
            "triangle": 0,
            "vertex": 0,
            "time": 0.0,
        }
        self.__clear_color = True
        self.__clear_depth = True
        self.__material_index = {}
        self.__pipelines = {}
        self.__meshes = {}
        self.__depth_texture = None
        self.__texture_atlas = {}
        self.__texture_ids = []
        self.__texture_names: list[str | None] = [None] * texture_array_size[2]

        self.__lights_buffer = (
            BufferBuilder()
            .with_size(light_buffer_size * 4 * 4 * 2)
            .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_DST)
            .build()
        )

        self.__camera_buffer = (
            BufferBuilder()
            .with_size(4 * 4 * 4 + 4 * 4)
            .with_usage(BufferUsage.UNIFORM | BufferUsage.COPY_DST)
            .build()
        )

    def create_pipeline(self, material_class: Type[Material]):
        if material_class not in self.__pipelines:
            shader_source = SHADER_START + material_class.get_fragment()
            pipeline = AutoRenderPipeline(shader_source).with_depth_texture()
            pipeline.set_vertex_buffer_descriptors(self.__vertex_buffer_descriptors)
            for i in range(len(self.__vertex_buffers)):
                pipeline.set_vertex_buffer(i, self.__vertex_buffers[i])
            pipeline.set_index_buffer(self.__index_buffer)
            pipeline.set_binding_buffer(0, 0, self.__camera_buffer)
            pipeline.set_binding_buffer(0, 1, self.__lights_buffer)
            material_class.set_bindings(
                pipeline,
                self.__material_buffer,
                self.__texture_ids_buffer,
                self.__textures,
            )
            self.__pipelines[material_class] = pipeline

    def begin_frame(self):
        self.__meshes = {}
        self.__start_time = time.perf_counter()

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

        if len(vertex_data[0]) > self.__vertex_buffer_size:
            raise IndexError("Vertex Buffer Not Long Enough")

        if len(index_data) > self.__index_buffer_size:
            raise IndexError("index Buffer Not Long Enough")

        if (
            self.__vertex_count + len(vertex_data[0]) > self.__vertex_buffer_size
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
                    if len(img) == 0:
                        self.__texture_ids.append(-1)
                    else:
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

        vertex_data = vertex_data + (
            array.from_numbers(float32, material_index).repeat(len(vertex_data[0])),
        )

        index_data = index_data + self.__vertex_count

        for i in range(len(self.__vertex_buffers)):
            buffer_offset = self.__vertex_count * VERTEX_DATA_SIZE[i]
            write_buffer(self.__vertex_buffers[i], vertex_data[i], buffer_offset)

        buffer_offset = self.__index_count * 4
        write_buffer(self.__index_buffer, index_data, buffer_offset)

        self.__vertex_count += len(vertex_data[0])
        self.__index_count += len(index_data)
        self.__frame_mesh_count += 1
        self.__frame_triangle_count += len(index_data) // 3
        self.__frame_vertex_count += len(vertex_data[0])

    def end_frame(
        self,
        output_texture: GPUTexture,
        camera: Camera,
        lights: array[vec4],
        clear_color=True,
        clear_depth=True,
    ):
        self.__clear_color = clear_color
        self.__clear_depth = clear_depth
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
        camera_position = vec4(camera.get_position(), 1.0)  # type: ignore
        camera_matrix = proj_matrix * view_matrix

        camera_data = camera_matrix.to_bytes() + camera_position.to_bytes()

        write_buffer(self.__camera_buffer, camera_data)

        write_buffer(self.__lights_buffer, lights)

        for material_class in self.__meshes:
            pipeline = self.__pipelines[material_class]
            pipeline.set_depth_texture(self.__depth_texture)
            for mesh, transform, material in self.__meshes[material_class]:
                self.__add_mesh(pipeline, output_texture, mesh, transform, material)
            self.__draw(pipeline, output_texture)

        if self.__clear_color or self.__clear_depth:
            self.clear(output_texture, self.__clear_color, self.__clear_depth)

        render_time = time.perf_counter() - self.__start_time

        self.__frame_stat = {
            "draw": self.__frame_draw_count,
            "mesh": self.__frame_mesh_count,
            "triangle": self.__frame_triangle_count,
            "vertex": self.__frame_vertex_count,
            "time": render_time,
        }

    def __draw(self, pipeline: AutoRenderPipeline, output_texture: GPUTexture):
        if len(self.__texture_ids) > 0:
            write_buffer(
                self.__texture_ids_buffer,
                array.from_numbers(int32, *self.__texture_ids),
            )

        pipeline.render(
            output_texture,
            self.__index_count,
            clear_color=self.__clear_color,
            clear_depth=self.__clear_depth,
        )

        self.__clear_color = False
        self.__clear_depth = False
        self.__frame_draw_count += 1
        self.__vertex_count = 0
        self.__index_count = 0
        self.__material_index = {}
        self.__texture_ids = []

    def clear(
        self,
        output_texture: GPUTexture,
        clear_color: bool = True,
        clear_depth: bool = True,
    ):
        cmd_bfr_bld = CommandBufferBuilder()
        rnd_pass_bld = cmd_bfr_bld.begin_render_pass(output_texture)
        rnd_pass_bld.with_load_op(LoadOp.clear if clear_color else LoadOp.load)
        if self.__depth_texture is not None:
            rnd_pass_bld.with_depth_stencil(self.__depth_texture, clear_depth)

        rnd_pass_bld.build().end()
        cmd_bfr_bld.submit()

    def get_frame_stat(self) -> dict:
        return self.__frame_stat
