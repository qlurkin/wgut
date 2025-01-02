from wgpu import BufferUsage, GPUTexture, IndexFormat, VertexFormat
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
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer


class MyApp(Window):
    def on_resize(self):
        canvas = self.get_canvas()
        self.depth_texture = TextureBuilder().build_depth(canvas.get_physical_size())
        width, height = self.get_canvas().get_logical_size()

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
            GraphicPipelineBuilder(self.get_texture_format())
            .with_shader("cube.wgsl")
            .with_depth_stencil()
            .with_vertex_buffer()
            .with_attribute(VertexFormat.float32x3)
            .with_attribute(VertexFormat.float32x3)
            .build()
        )

        self.on_resize()

        self.imgui_renderer = ImguiRenderer(
            get_device(), self.get_canvas(), self.get_texture_format()
        )

        self.imgui_renderer.set_gui(self.gui)

        self.frame_time = 0

    def update(self, delta_time: float):
        self.imgui_renderer.backend.io.delta_time = delta_time
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

        command_buffer_builder.submit()

        self.imgui_renderer.render()

    def gui(self) -> imgui.ImDrawData:
        imgui.new_frame()
        imgui.begin("Hello Imgui", None)
        imgui.text(f"Frame Time: {self.frame_time:.5f}s")
        imgui.end()
        imgui.end_frame()
        imgui.render()
        return imgui.get_draw_data()

    def process_event(self, event):
        if event["event_type"] == "resize":
            self.on_resize()


MyApp().run()
