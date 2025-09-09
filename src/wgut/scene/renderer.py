from numpy.typing import NDArray
from wgpu import (
    AddressMode,
    BufferBindingType,
    BufferUsage,
    CompareFunction,
    CullMode,
    FilterMode,
    FrontFace,
    GPURenderPassEncoder,
    GPUTexture,
    IndexFormat,
    LoadOp,
    MipmapFilterMode,
    PrimitiveTopology,
    ShaderStage,
    StoreOp,
    TextureFormat,
    TextureUsage,
)

from wgut.scene.mesh import Mesh, get_vertex_buffer_descriptors
from wgut.scene.transform import Transform
from ..core import get_device, load_image, write_buffer, write_texture
import time
import numpy as np
from pathlib import Path
import re

shader_filepath = Path(__file__).parent / "renderer_shader.wgsl"

with open(shader_filepath, encoding="utf8") as file:
    SHADER = file.read()

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
VERTEX_TOTAL_SIZE = sum(VERTEX_DATA_SIZE.values())


class Material:
    def __init__(
        self,
        albedo: str | tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        normal: str | tuple[float, float, float] = (0.5, 0.5, 1.0),
        roughness: str | float = 0.5,
        metalicity: str | float = 0.0,
        emissivity: str | tuple[float, float, float] = (0.0, 0.0, 0.0),
        occlusion: str | float = 1.0,
    ):
        self.__texture_count = 0
        self.__has_none_texture = False

        self.__albedo_texture, self.__albedo_value = self.__init_material_property(
            albedo, [1, 1, 1, 1]
        )
        self.__normal_texture, self.__normal_value = self.__init_material_property(
            normal, [1, 1, 1]
        )
        self.__roughness_texture, self.__roughness_value = (
            self.__init_material_property(roughness, 1)
        )
        self.__metalicity_texture, self.__metalicity_value = (
            self.__init_material_property(metalicity, 1)
        )
        self.__emissivity_texture, self.__emissivity_value = (
            self.__init_material_property(emissivity, [1, 1, 1])
        )
        self.__occlusion_texture, _ = self.__init_material_property(occlusion, 1)

        if self.__has_none_texture:
            self.__texture_count += 1

    def __init_material_property(self, texture, value) -> tuple[str, NDArray]:
        if isinstance(texture, str):
            return texture, np.array(value, dtype=np.float32)
        else:
            self.__has_none_texture = True
            return "", np.array(texture, dtype=np.float32)

    @staticmethod
    def get_data_size() -> int:
        return 17 * 4 + 12

    def get_texture_count(self) -> int:
        return self.__texture_count

    def get_textures(self) -> list[str]:
        return [
            self.__albedo_texture,
            self.__normal_texture,
            self.__roughness_texture,
            self.__metalicity_texture,
            self.__emissivity_texture,
            self.__occlusion_texture,
        ]

    def get_data(self, tex_ids: list[int]) -> bytes:
        res = b""
        res += self.__albedo_value.tobytes()
        res += self.__emissivity_value.tobytes()
        res += np.int32(tex_ids[4]).tobytes()
        res += self.__normal_value.tobytes()
        res += np.int32(tex_ids[1]).tobytes()
        res += self.__roughness_value.tobytes()
        res += np.int32(tex_ids[2]).tobytes()
        res += self.__metalicity_value.tobytes()
        res += np.int32(tex_ids[3]).tobytes()
        res += np.int32(tex_ids[5]).tobytes()
        res += np.int32(tex_ids[0]).tobytes()

        return res


class Renderer:
    def __init__(
        self,
        vertex_batch_count: int,
        index_batch_count: int,
        max_light_count: int,
        max_material_count: int,
        max_texture_count: int,
    ):
        self.__batch_max_vertex_count = vertex_batch_count
        self.__batch_max_index_count = index_batch_count
        self.__max_light_count = max_light_count
        self.__max_material_count = max_material_count
        self.__batch_max_texture_count = max_texture_count

        self.__vertex_size = VERTEX_TOTAL_SIZE
        self.__index_size = 4
        self.__light_size = 32
        self.__material_size = Material.get_data_size()

        self.__vertex_buffer = get_device().create_buffer(
            label="Renderer Vertex Buffer",
            size=self.__vertex_size * self.__batch_max_vertex_count,
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,  # type: ignore
        )

        self.__index_buffer = get_device().create_buffer(
            label="Renderer Index Buffer",
            size=self.__index_size * self.__batch_max_index_count,
            usage=BufferUsage.INDEX | BufferUsage.COPY_DST,  # type: ignore
        )

        self.__lights_buffer = get_device().create_buffer(
            label="Renderer Lights Buffer",
            size=self.__light_size * self.__max_light_count,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,  # type: ignore
        )

        self.__lights_count_buffer = get_device().create_buffer(
            label="Renderer Lights Count Buffer",
            size=4,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,  # type: ignore
        )

        self.__camera_buffer = get_device().create_buffer(
            label="Renderer Camera Buffer",
            size=4 * 4 * 4 + 4 * 4,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,  # type: ignore
        )

        self.__material_data_buffer = get_device().create_buffer(
            label="Renderer Material Data Buffer",
            size=self.__max_material_count * self.__material_size,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,  # type: ignore
        )

        self.__texture_cache = {}

        white_texture = get_device().create_texture(
            size=[1, 1],
            format=TextureFormat.rgba8unorm,  # type: ignore
            usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,  # type: ignore
        )
        write_texture(white_texture, np.array([[[255, 255, 255, 255]]], dtype=np.uint8))

        self.__texture_cache[""] = white_texture

        self.__vertex_buffer_descriptors = get_vertex_buffer_descriptors()

        shader_textures_bindings = "@group(1) @binding(0) var samplr: sampler;\n"
        for i in range(1, self.__batch_max_texture_count + 1):
            shader_textures_bindings += (
                f"@group(1) @binding({i}) var texture{i}: texture_2d<f32>;\n"
            )

        shader_sample_function = ""
        for i in range(1, self.__batch_max_texture_count + 1):
            shader_sample_function += f"    if(id == {i}) {{ return textureSample(texture{i}, samplr, uv); }}\n"
        shader_sample_function = f"fn sample(id: i32, uv: vec2<f32>) -> vec4<f32> {{\n{shader_sample_function}    return vec4<f32>(0.0, 0.0, 0.0, 1.0);\n}}\n"

        shader_source = re.sub(
            r"// TERTURES_INJECTION_START.*// TEXTURE_INJECTION_END",
            shader_textures_bindings + shader_sample_function,
            SHADER,
            flags=re.DOTALL,
        )

        self.__shader_module = get_device().create_shader_module(code=shader_source)

        bind_group_layout_0 = get_device().create_bind_group_layout(
            label="Bind Group 0 Layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": ShaderStage.VERTEX | ShaderStage.FRAGMENT,
                    "buffer": {"type": BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": ShaderStage.FRAGMENT,
                    "buffer": {"type": BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": ShaderStage.FRAGMENT,
                    "buffer": {"type": BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": ShaderStage.FRAGMENT,
                    "buffer": {"type": BufferBindingType.read_only_storage},
                },
            ],
        )

        self.__bind_group_0 = get_device().create_bind_group(
            layout=bind_group_layout_0,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.__camera_buffer,
                        "offset": 0,
                        "size": self.__camera_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.__lights_buffer,
                        "offset": 0,
                        "size": self.__lights_buffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.__lights_count_buffer,
                        "offset": 0,
                        "size": self.__lights_count_buffer.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.__material_data_buffer,
                        "offset": 0,
                        "size": self.__material_data_buffer.size,
                    },
                },
            ],
        )

        group1_enytries = [
            {
                "binding": 0,
                "visibility": ShaderStage.FRAGMENT,
                "sampler": {},
            }
        ]
        for i in range(1, self.__batch_max_texture_count + 1):
            group1_enytries.append(
                {
                    "binding": i,
                    "visibility": ShaderStage.FRAGMENT,
                    "texture": {},
                }
            )

        self.__bind_group_layout_1 = get_device().create_bind_group_layout(
            label="Bind Group 1 Layout",
            entries=group1_enytries,
        )

        self.__sampler = get_device().create_sampler(
            address_mode_u=AddressMode.repeat,  # type: ignore
            address_mode_v=AddressMode.repeat,  # type: ignore
            address_mode_w=AddressMode.repeat,  # type: ignore
            mag_filter=FilterMode.nearest,  # type: ignore
            min_filter=FilterMode.nearest,  # type: ignore
            mipmap_filter=MipmapFilterMode.nearest,  # type: ignore
        )

        bind_group_layouts = [bind_group_layout_0, self.__bind_group_layout_1]

        self.__pipeline_layout = get_device().create_pipeline_layout(
            label="Renderer Pipeline Layout",
            bind_group_layouts=bind_group_layouts,
        )

        self.__pipeline = None

        self.__depth_texture: GPUTexture | None = None

    def __begin_batch(self):
        self.__flush()

    def __batch(self, mesh: Mesh, transform: Transform, material: Material):
        vertex_data = mesh.get_transformed_vertices(transform.get_matrix())
        index_data = mesh.get_indices()

        if len(vertex_data) > self.__batch_max_vertex_count:
            raise IndexError("Vertex Buffer Not Long Enough")

        if len(index_data) > self.__batch_max_index_count:
            raise IndexError("Index Buffer Not Long Enough")

        material_ok = True
        textures = []

        for t in material.get_textures():
            if t not in self.__texture_cache:
                image = load_image(t)
                texture = get_device().create_texture(
                    size=image.size,
                    format=TextureFormat.rgba8unorm,  # type: ignore
                    usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,  # type: ignore
                )
                write_texture(texture, image)
                self.__texture_cache[t] = texture
            textures.append(self.__texture_cache[t])

        if material not in self.__current_batch_materials_index:
            new_textures_count = 0
            for texture in textures:
                if texture not in self.__current_batch_textures_index:
                    new_textures_count += 1
            if (
                len(self.__current_batch_textures_index) + new_textures_count
                > self.__batch_max_texture_count
            ):
                material_ok = False
            if (
                len(self.__current_batch_materials_index) + 1
            ) * self.__material_size > self.__max_material_count:
                material_ok = False

        if (
            self.__vertex_count + len(vertex_data) > self.__batch_max_vertex_count
            or self.__index_count + len(index_data) > self.__batch_max_index_count
            or not material_ok
        ):
            self.__draw()

        if material not in self.__current_batch_materials_index:
            mat_id = len(self.__current_batch_materials_index)
            tex_ids = []
            for texture in textures:
                if texture not in self.__current_batch_textures_index:
                    tex_id = (
                        len(self.__current_batch_textures) + 1
                    )  # tex_ids are 1 indexed
                    self.__current_batch_textures_index[texture] = tex_id
                    self.__current_batch_textures.append(texture)
                tex_ids.append(self.__current_batch_textures_index[texture])

            mat_data = material.get_data(tex_ids)

            write_buffer(
                self.__material_data_buffer,
                mat_data,
                mat_id * material.get_data_size(),
            )

            self.__current_batch_materials_index[material] = mat_id
        mat_id = self.__current_batch_materials_index[material]

        vertex_data = np.hstack(
            [
                vertex_data,
                np.full((len(vertex_data), 1), mat_id, dtype=np.float32),
            ]
        )
        write_buffer(
            buffer=self.__vertex_buffer,
            data=vertex_data,
            buffer_offset=self.__vertex_count * self.__vertex_size,
        )

        index_data = index_data + self.__vertex_count
        write_buffer(
            buffer=self.__index_buffer,
            data=index_data,
            buffer_offset=self.__index_count * self.__index_size,
        )

        self.__vertex_count += len(vertex_data)
        self.__index_count += len(index_data)

        self.__frame_mesh_count += 1
        self.__frame_triangle_count += len(index_data) // 3
        self.__frame_vertex_count += len(vertex_data)

    def __end_batch(self):
        if self.__vertex_count > 0:
            self.__draw()

    def __draw(self):
        assert self.__depth_texture is not None, "`begin_frame()` not called ?"
        assert self.__pipeline is not None, "`begin_frame()` not called ?"

        command_encoder = get_device().create_command_encoder()

        render_pass: GPURenderPassEncoder = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.__current_frame_output_texture.create_view(),
                    "resolve_target": None,
                    "load_op": LoadOp.load,
                    "store_op": StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.__depth_texture.create_view(),
                "depth_load_op": LoadOp.load,
                "depth_store_op": StoreOp.store,
            },
        )

        group_1_entries = [
            {
                "binding": 0,
                "resource": self.__sampler,
            }
        ]

        for i in range(len(self.__current_batch_textures)):
            group_1_entries.append(
                {
                    "binding": i + 1,
                    "resource": self.__current_batch_textures[i].create_view(),
                }
            )

        for i in range(
            len(self.__current_batch_textures), self.__batch_max_texture_count
        ):
            group_1_entries.append(
                {
                    "binding": i + 1,
                    "resource": self.__texture_cache[""].create_view(),
                }
            )

        bind_group_1 = get_device().create_bind_group(
            layout=self.__bind_group_layout_1,
            entries=group_1_entries,
        )

        render_pass.set_pipeline(self.__pipeline)
        render_pass.set_bind_group(0, self.__bind_group_0)
        render_pass.set_bind_group(1, bind_group_1)
        render_pass.set_vertex_buffer(0, self.__vertex_buffer)
        render_pass.set_index_buffer(self.__index_buffer, IndexFormat.uint32)  # type: ignore
        render_pass.draw_indexed(self.__index_count)
        render_pass.end()

        get_device().queue.submit([command_encoder.finish()])
        self.__frame_draw_count += 1

        self.__flush()

    def __flush(self):
        self.__current_batch_materials_index = {}
        self.__current_batch_textures = []
        self.__current_batch_textures_index = {}
        self.__vertex_count = 0
        self.__index_count = 0

    def begin_frame(
        self,
        output_texture: GPUTexture,
        camera_matrix: NDArray,
        camera_position: NDArray,
        lights: NDArray | None,
    ):
        self.__current_frame_output_texture = output_texture
        self.__current_frame_camera_matrix = camera_matrix
        self.__current_frame_camera_position = camera_position
        self.__current_frame_lights = lights

        if self.__pipeline is None:
            self.__pipeline = self.__create_pipeline(output_texture.format)

        self.__start_time = time.perf_counter()
        self.__meshes = []

        self.__frame_stat = {
            "draw": 0,
            "mesh": 0,
            "triangle": 0,
            "vertex": 0,
            "time": 0.0,
        }

    def add_mesh(self, mesh: Mesh, transform: Transform, material: Material):
        if material.get_texture_count() > self.__batch_max_texture_count:
            raise IndexError("Not enough texture's slots")
        self.__meshes.append((mesh, transform, material))

    def end_frame(
        self,
    ):
        self.__frame_draw_count = 0
        self.__frame_mesh_count = 0
        self.__frame_triangle_count = 0
        self.__frame_vertex_count = 0

        if (
            self.__depth_texture is None
            or self.__depth_texture.size != self.__current_frame_output_texture.size
        ):
            self.__depth_texture = get_device().create_texture(
                label="Renderer Depth Texture",
                size=self.__current_frame_output_texture.size,
                format=TextureFormat.depth32float,  # type: ignore
                usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING,  # type: ignore
            )

        # Must send transpose version of matrices, because GPU expect matrices in column major order
        camera_data = np.vstack(
            [
                np.ascontiguousarray(self.__current_frame_camera_matrix.T),
                self.__current_frame_camera_position,
            ]
        )
        write_buffer(buffer=self.__camera_buffer, data=camera_data)

        light_count = 0
        if self.__current_frame_lights is not None:
            write_buffer(buffer=self.__lights_buffer, data=self.__current_frame_lights)
            light_count = len(self.__current_frame_lights)
        write_buffer(
            buffer=self.__lights_count_buffer,
            data=np.array([light_count], dtype=np.int32),
        )

        self.__begin_batch()
        for mesh, transform, material in self.__meshes:
            self.__batch(mesh, transform, material)
        self.__end_batch()

        self.__frame_stat = {
            "draw": self.__frame_draw_count,
            "mesh": self.__frame_mesh_count,
            "triangle": self.__frame_triangle_count,
            "vertex": self.__frame_vertex_count,
            "time": time.perf_counter() - self.__start_time,
        }

    def __create_pipeline(
        self,
        output_format: TextureFormat,
    ):
        return get_device().create_render_pipeline(
            label="Renderer Pipeline",
            layout=self.__pipeline_layout,
            vertex={
                "module": self.__shader_module,
                "entry_point": "vs_main",
                "buffers": self.__vertex_buffer_descriptors,
            },
            primitive={
                "topology": PrimitiveTopology.triangle_list,
                "front_face": FrontFace.ccw,
                "cull_mode": CullMode.back,
            },
            depth_stencil={
                "format": TextureFormat.depth32float,
                "depth_write_enabled": True,
                "depth_compare": CompareFunction.less,
            },
            multisample=None,
            fragment={
                "module": self.__shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": output_format,
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                ],
            },
        )

    def clear_color(
        self, output_texture: GPUTexture, clear_color: tuple[float, float, float, float]
    ):
        command_encoder = get_device().create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": output_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": clear_color,
                    "load_op": LoadOp.clear,
                    "store_op": StoreOp.store,
                }
            ],
        )
        render_pass.end()

        get_device().queue.submit([command_encoder.finish()])

    def clear_depth(self):
        if self.__depth_texture is None:
            return

        command_encoder = get_device().create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": self.__depth_texture.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": LoadOp.clear,
                "depth_store_op": StoreOp.store,
            },
        )
        render_pass.end()

        get_device().queue.submit([command_encoder.finish()])

    def get_frame_stat(self) -> dict:
        return self.__frame_stat
