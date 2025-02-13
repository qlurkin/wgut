from wgpu import BufferUsage, GPUTexture, IndexFormat, ShaderStage, VertexFormat
from wgut.builders import (
    BindGroupBuilder,
    BingGroupLayoutBuilder,
    RenderPipelineBuilder,
    VertexBufferDescriptorsBuilder,
    BufferBuilder,
    CommandBufferBuilder,
    PipelineLayoutBuilder,
    TextureBuilder,
    write_buffer,
)
from wgut.window import Window
import numpy as np
import wgut.cgmath as cm


class MyApp(Window):
    def on_resize(self):
        canvas = self.get_canvas()
        self.depth_texture = TextureBuilder().build_depth(canvas.get_physical_size())

        width, height = canvas.get_logical_size()

        view_matrix = cm.look_at([3, 2, 4], [0, 0, 0], [0, 1, 0])
        proj_matrix = cm.perspective(45, width / height, 0.1, 100)

        camera_data = np.array([view_matrix, proj_matrix])

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

        self.vertex_buffer = (
            BufferBuilder()
            .from_data(vertex_data)
            .with_usage(BufferUsage.VERTEX)
            .build()
        )

        self.index_buffer = (
            BufferBuilder().from_data(index_data).with_usage(BufferUsage.INDEX).build()
        )

        bg_layout = BingGroupLayoutBuilder().with_buffer(ShaderStage.VERTEX).build()

        p_layout = PipelineLayoutBuilder().with_bind_group_layout(bg_layout).build()

        self.camera_buffer = (
            BufferBuilder()
            .with_size(2 * 4 * 4 * 4)
            .with_usage(BufferUsage.UNIFORM | BufferUsage.COPY_DST)
            .build()
        )

        self.camera_bind_group = (
            BindGroupBuilder(bg_layout).with_buffer(self.camera_buffer).build()
        )

        vertex_buffer_descriptors = (
            VertexBufferDescriptorsBuilder()
            .with_vertex_buffer()
            .with_attribute(VertexFormat.float32x3)
            .with_attribute(VertexFormat.float32x3)
            .build()
        )

        self.pipeline = (
            RenderPipelineBuilder(self.get_texture_format(), vertex_buffer_descriptors)
            .with_layout(p_layout)
            .with_shader("cube.wgsl")
            .with_depth_stencil()
            .build()
        )

        self.on_resize()

    def render(self, screen: GPUTexture):
        command_buffer_builder = CommandBufferBuilder()

        render_pass = (
            command_buffer_builder.begin_render_pass(screen)
            .with_depth_stencil(self.depth_texture)
            .build()
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_index_buffer(self.index_buffer, IndexFormat.uint32)  # type: ignore
        render_pass.set_bind_group(0, self.camera_bind_group)
        render_pass.draw_indexed(36)
        render_pass.end()

        command_buffer_builder.submit()

    def process_event(self, event):
        if event["event_type"] == "resize":
            self.on_resize()


MyApp().run()
