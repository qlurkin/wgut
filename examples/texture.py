from wgpu import (
    AddressMode,
    BufferUsage,
    CullMode,
    FilterMode,
    FrontFace,
    GPURenderPassEncoder,
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
    Window,
    load_file,
    get_device,
    load_image,
    submit_command,
    write_buffer,
    write_texture,
    create_canvas,
)
import numpy as np


canvas = create_canvas()


class MyApp(Window):
    def setup(self):
        self.set_title("Hello Triangle")
        vertex_data = np.array(
            [
                # x, y, u, v
                [0.0, 0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.0, 1.0],
                [0.5, -0.5, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.vertex_buffer = get_device().create_buffer(
            size=vertex_data.nbytes,
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,  # type: ignore
        )
        write_buffer(self.vertex_buffer, vertex_data)

        vertex_buffer_descriptor = {
            "array_stride": 4 * 4,
            "step_mode": VertexStepMode.vertex,
            "attributes": [
                {
                    "format": VertexFormat.float32x2,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": VertexFormat.float32x2,
                    "offset": 2 * 4,
                    "shader_location": 1,
                },
            ],
        }

        shader_module = get_device().create_shader_module(
            code=load_file("texture.wgsl")
        )

        bind_group_layout = get_device().create_bind_group_layout(
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

        pipeline_layout = get_device().create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        self.pipeline = get_device().create_render_pipeline(
            label="Render Pipeline",
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
            depth_stencil=None,
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

        sampler = get_device().create_sampler(
            address_mode_u=AddressMode.repeat,  # type: ignore
            address_mode_v=AddressMode.repeat,  # type: ignore
            mag_filter=FilterMode.nearest,  # type: ignore
            min_filter=FilterMode.nearest,  # type: ignore
            mipmap_filter=MipmapFilterMode.nearest,  # type: ignore
        )

        image = load_image("./textures/Wood_025_basecolor.jpg")
        texture = get_device().create_texture(
            size=image.size,
            format=TextureFormat.rgba8unorm_srgb,  # type: ignore
            usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,  # type: ignore
        )
        write_texture(texture, image)

        self.bind_group = get_device().create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": texture.create_view(),
                },
                {"binding": 1, "resource": sampler},
            ],
        )

    def render(self):
        command_encoder = get_device().create_command_encoder()

        render_pass: GPURenderPassEncoder = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.get_current_texture().create_view(),
                    "resolve_target": None,
                    "clear_value": (0.9, 0.9, 0.9, 1.0),
                    "load_op": LoadOp.clear,
                    "store_op": StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.draw(3)
        render_pass.end()

        submit_command(command_encoder)


MyApp(canvas).run()
