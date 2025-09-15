# imgui imports first
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from wgpu import (
    BufferBindingType,
    BufferUsage,
    CompareFunction,
    CullMode,
    FrontFace,
    GPURenderPassEncoder,
    GPUTexture,
    IndexFormat,
    LoadOp,
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
import wgut.cgmath as cm
from wgut.core import get_device, submit_command


class MyApp(Window):
    def on_resize(self):
        canvas = self.get_canvas()
        self.depth_texture = get_device().create_texture(
            size=canvas.get_physical_size(),
            format=TextureFormat.depth32float,  # type: ignore
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING,  # type: ignore
        )

        width, height = canvas.get_physical_size()  # type: ignore

        view_matrix = cm.look_at([3, 2, 4], [0, 0, 0], [0, 1, 0])
        proj_matrix = cm.perspective(45, width / height, 0.1, 100)

        # Must send transpose version of matrices, because GPU expect matrices in column major order
        camera_data = np.array([view_matrix.T, proj_matrix.T])

        write_buffer(self.camera_buffer, camera_data)

    def setup(self):
        self.set_title("Hello Cube")

        # fmt: off
        vertex_data = np.array(
            [
                # x, y, z, r, g, b
                [0.5, 0.5, 0.5, 1.0, 0.0, 0.0],
                [-0.5, 0.5, 0.5, 1.0, 0.0, 0.0],
                [-0.5, -0.5, 0.5, 1.0, 0.0, 0.0],
                [0.5, -0.5, 0.5, 1.0, 0.0, 0.0],

                [0.5, 0.5, 0.5, 0.0, 1.0, 0.0],
                [0.5, -0.5, 0.5, 0.0, 1.0, 0.0],
                [0.5, -0.5, -0.5, 0.0, 1.0, 0.0],
                [0.5, 0.5, -0.5, 0.0, 1.0, 0.0],

                [0.5, 0.5, -0.5, 0.0, 0.0, 1.0],
                [0.5, -0.5, -0.5, 0.0, 0.0, 1.0],
                [-0.5, -0.5, -0.5, 0.0, 0.0, 1.0],
                [-0.5, 0.5, -0.5, 0.0, 0.0, 1.0],

                [-0.5, 0.5, 0.5, 1.0, 1.0, 0.0],
                [-0.5, 0.5, -0.5, 1.0, 1.0, 0.0],
                [-0.5, -0.5, -0.5, 1.0, 1.0, 0.0],
                [-0.5, -0.5, 0.5, 1.0, 1.0, 0.0],

                [0.5, 0.5, 0.5, 0.0, 1.0, 1.0],
                [0.5, 0.5, -0.5, 0.0, 1.0, 1.0],
                [-0.5, 0.5, -0.5, 0.0, 1.0, 1.0],
                [-0.5, 0.5, 0.5, 0.0, 1.0, 1.0],
    
                [0.5, -0.5, 0.5, 1.0, 0.0, 1.0],
                [-0.5, -0.5, 0.5, 1.0, 0.0, 1.0],
                [-0.5, -0.5, -0.5, 1.0, 0.0, 1.0],
                [0.5, -0.5, -0.5, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        index_data = np.array(
            [
                 0,  1,  2,  0,  2,  3,
                 4,  5,  6,  4,  6,  7,
                 8,  9, 10,  8, 10, 11,
                12, 13, 14, 12, 14, 15,
                16, 17, 18, 16, 18, 19,
                20, 21, 22, 20, 22, 23,
            ],
            dtype=np.uint32,
        )
        # fmt: on

        self.vertex_buffer = get_device().create_buffer(
            size=vertex_data.nbytes,
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,  # type: ignore
        )
        write_buffer(self.vertex_buffer, vertex_data)

        self.index_buffer = get_device().create_buffer(
            size=index_data.nbytes,
            usage=BufferUsage.INDEX | BufferUsage.COPY_DST,  # type: ignore
        )
        write_buffer(self.index_buffer, index_data)

        bg_layout = get_device().create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": ShaderStage.VERTEX,
                    "buffer": {"type": BufferBindingType.uniform},
                },
            ]
        )

        p_layout = get_device().create_pipeline_layout(bind_group_layouts=[bg_layout])

        self.camera_buffer = get_device().create_buffer(
            size=2 * 4 * 4 * 4,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,  # type: ignore
        )

        self.camera_bind_group = get_device().create_bind_group(
            layout=bg_layout,
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
            "array_stride": 6 * 4,
            "step_mode": VertexStepMode.vertex,
            "attributes": [
                {
                    "format": VertexFormat.float32x3,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": VertexFormat.float32x3,
                    "offset": 3 * 4,
                    "shader_location": 1,
                },
            ],
        }

        shader_module = get_device().create_shader_module(code=load_file("./cube.wgsl"))

        self.pipeline = get_device().create_render_pipeline(
            layout=p_layout,
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

        self.imgui_renderer = ImguiRenderer(
            get_device(), self.get_canvas(), self.get_texture_format()
        )

        self.imgui_renderer.set_gui(self.gui)

        self.frame_time = 0

    def update(self, delta_time: float):
        self.frame_time = delta_time

    def render(self, screen: GPUTexture):
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
        render_pass.draw_indexed(36)
        render_pass.end()

        submit_command(command_encoder)

        self.imgui_renderer.render()

    def gui(self):
        imgui.begin("Hello Imgui", None)
        imgui.text(f"Frame Time: {self.frame_time:.5f}s")
        imgui.end()

    def process_event(self, event):
        if event["event_type"] == "resize":
            self.on_resize()


MyApp().run()
