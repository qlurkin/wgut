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
    VertexFormat,
    VertexStepMode,
)
from wgut import (
    write_buffer,
    Window,
    load_file,
)
import numpy as np
from wgut.core import get_device, load_image, submit_command, write_texture
from wgut.orbit_camera import OrbitCamera
from wgut.scene.primitives.icosphere import icosphere_with_uv


class MyApp(Window):
    def on_resize(self):
        canvas = self.get_canvas()
        self.depth_texture = get_device().create_texture(
            size=canvas.get_physical_size(),
            format=TextureFormat.depth32float,  # type: ignore
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING,  # type: ignore
        )

    def setup(self):
        self.set_title("Hello Icosphere")

        vertex_data_positions, vertex_data_uvs, indices = icosphere_with_uv(3)
        vertex_data = np.hstack([vertex_data_positions, vertex_data_uvs])

        self.camera = OrbitCamera((0, 2, -4), (0, 0, 0), 45, 0.1, 100)

        self.vertex_buffer = get_device().create_buffer(
            size=vertex_data.nbytes,
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,  # type: ignore
        )
        write_buffer(self.vertex_buffer, vertex_data)

        self.index_buffer = get_device().create_buffer(
            size=indices.nbytes,
            usage=BufferUsage.INDEX | BufferUsage.COPY_DST,  # type: ignore
        )
        write_buffer(self.index_buffer, indices)

        self.indices_count = len(indices)

        camera_bind_group_layout = get_device().create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": ShaderStage.VERTEX,
                    "buffer": {"type": BufferBindingType.uniform},
                },
            ]
        )

        self.camera_buffer = get_device().create_buffer(
            size=2 * 4 * 4 * 4,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,  # type: ignore
        )

        self.camera_bind_group = get_device().create_bind_group(
            layout=camera_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.camera_buffer,
                        "offset": 0,
                        "size": self.camera_buffer.size,
                    },
                },
            ],
        )

        vertex_buffer_descriptor = {
            "array_stride": 5 * 4,
            "step_mode": VertexStepMode.vertex,
            "attributes": [
                {
                    "format": VertexFormat.float32x3,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": VertexFormat.float32x2,
                    "offset": 3 * 4,
                    "shader_location": 1,
                },
            ],
        }

        shader_module = get_device().create_shader_module(
            code=load_file("./textured_icosphere.wgsl")
        )

        image = load_image("./textures/Wood_025_basecolor.jpg")
        diffuse_texture = get_device().create_texture(
            size=image.size,
            format=TextureFormat.rgba8unorm_srgb,  # type: ignore
            usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,  # type: ignore
        )
        write_texture(diffuse_texture, image)

        sampler = get_device().create_sampler(
            address_mode_u=AddressMode.repeat,  # type: ignore
            address_mode_v=AddressMode.repeat,  # type: ignore
            mag_filter=FilterMode.nearest,  # type: ignore
            min_filter=FilterMode.nearest,  # type: ignore
            mipmap_filter=MipmapFilterMode.nearest,  # type: ignore
        )

        texture_bind_group_layout = get_device().create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": ShaderStage.FRAGMENT,
                    "texture": {},
                },
                {
                    "binding": 1,
                    "visibility": ShaderStage.FRAGMENT,
                    "sampler": {},
                },
            ]
        )

        self.texture_bind_group = get_device().create_bind_group(
            layout=texture_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": diffuse_texture.create_view(),
                },
                {"binding": 1, "resource": sampler},
            ],
        )

        pipeline_layout = get_device().create_pipeline_layout(
            bind_group_layouts=[camera_bind_group_layout, texture_bind_group_layout]
        )

        self.pipeline = get_device().create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_buffer_descriptor],
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
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.get_texture_format(),
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                ],
            },
        )

        self.on_resize()

    def render(self, screen: GPUTexture):
        view_matrix, proj_matrix = self.camera.get_matrices(
            screen.width / screen.height
        )

        # Must send transpose version of matrices, because GPU expect matrices in column major order
        camera_data = np.array([view_matrix.T, proj_matrix.T])

        write_buffer(self.camera_buffer, camera_data)

        command_encoder = get_device().create_command_encoder()

        render_pass: GPURenderPassEncoder = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": screen.create_view(),
                    "resolve_target": None,
                    "clear_value": (0.9, 0.9, 0.9, 1.0),
                    "load_op": LoadOp.clear,
                    "store_op": StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_texture.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": LoadOp.clear,
                "depth_store_op": StoreOp.store,
            },
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_index_buffer(self.index_buffer, IndexFormat.uint32)  # type: ignore
        render_pass.set_bind_group(0, self.camera_bind_group)
        render_pass.set_bind_group(1, self.texture_bind_group)
        render_pass.draw_indexed(self.indices_count)
        render_pass.end()

        submit_command(command_encoder)

    def process_event(self, event):
        if event["event_type"] == "resize":
            self.on_resize()
        else:
            self.camera.process_event(event)


MyApp().run()
