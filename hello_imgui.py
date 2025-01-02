from wgpu import BufferUsage, GPUTexture, IndexFormat, LoadOp, VertexFormat
from gpu import (
    BindGroupBuilder,
    GraphicPipelineBuilder,
    BufferBuilder,
    CommandBufferBuilder,
    TextureBuilder,
    get_device,
)
from window import Window
import numpy as np
import cgmath as cm
from imgui_bundle import ImVec2, imgui
from wgpu.utils.imgui import ImguiWgpuBackend


class MyApp(Window):
    def on_resize(self):
        canvas = self.get_canvas()
        self.depth_texture = TextureBuilder().build_depth(canvas.get_physical_size())
        width, height = self.get_canvas().get_logical_size()
        self.imgui_backend.io.display_size = ImVec2(width, height)

        view_matrix = cm.look_at([3, 2, 4], [0, 0, 0], [0, 1, 0])
        proj_matrix = cm.perspective(45, width / height, 0.1, 100)

        camera_data = np.array([view_matrix, proj_matrix])

        self.camera_buffer = (
            BufferBuilder()
            .from_data(camera_data)
            .with_usage(BufferUsage.UNIFORM)
            .build()
        )

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

        self.vertex_buffer = (
            BufferBuilder()
            .from_data(vertex_data)
            .with_usage(BufferUsage.VERTEX)
            .build()
        )

        self.index_buffer = (
            BufferBuilder().from_data(index_data).with_usage(BufferUsage.INDEX).build()
        )

        self.pipeline = (
            GraphicPipelineBuilder()
            .with_shader("cube.wgsl")
            .with_depth_stencil()
            .with_vertex_buffer()
            .with_attribute(VertexFormat.float32x3)
            .with_attribute(VertexFormat.float32x3)
            .build()
        )

        imgui.create_context()
        self.imgui_backend = ImguiWgpuBackend(get_device(), self.get_texture_format())

        pixel_ratio = self.get_canvas().get_pixel_ratio()

        self.imgui_backend.io.display_framebuffer_scale = ImVec2(
            pixel_ratio, pixel_ratio
        )

        self.on_resize()
        self.frame_time = 0

    def update(self, delta_time: float):
        self.frame_time = delta_time

    def render(self, screen: GPUTexture):
        command_buffer_builder = CommandBufferBuilder()

        camera_bind_group = (
            BindGroupBuilder(self.pipeline.get_bind_group_layout(0))
            .with_buffer_binding(self.camera_buffer)
            .build()
        )

        render_pass = (
            command_buffer_builder.begin_render_pass(screen)
            .with_depth_stencil(self.depth_texture)
            .build()
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_index_buffer(self.index_buffer, IndexFormat.uint32)  # type: ignore
        render_pass.set_bind_group(0, camera_bind_group)
        render_pass.draw_indexed(36)
        render_pass.end()

        render_pass = (
            command_buffer_builder.begin_render_pass(screen)
            .with_load_op(LoadOp.load)
            .build()
        )
        imgui_data = self.gui()
        self.imgui_backend.render(imgui_data, render_pass)
        render_pass.end()

        command_buffer_builder.submit()

    def gui(self):
        imgui.new_frame()
        imgui.begin("Hello Imgui", None)
        imgui.text(f"Frame Time: {self.frame_time}")
        imgui.end()
        imgui.end_frame()
        imgui.render()
        return imgui.get_draw_data()

    def process_event(self, event):
        if event["event_type"] == "resize":
            self.on_resize()

        if event["event_type"] == "pointer_down" or event["event_type"] == "pointer_up":
            event_type = event["event_type"]
            down = event_type == "pointer_down"
            self.imgui_backend.io.add_mouse_button_event(event["button"] - 1, down)

        if event["event_type"] == "pointer_move":
            self.imgui_backend.io.add_mouse_pos_event(event["x"], event["y"])


MyApp().run()
