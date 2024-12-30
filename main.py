from wgpu import BufferUsage, GPUTexture, VertexFormat
from gpu import GraphicPipelineBuilder, BufferBuilder, CommandBufferBuilder
from window import Window, App
import numpy as np


class MyApp(App):
    def setup(self, size: tuple[int, int]):
        vertex_data = np.array(
            [
                # x, y, r, g, b
                [0.0, 0.5, 1.0, 0.0, 0.0],
                [-0.5, -0.5, 0.0, 1.0, 0.0],
                [0.5, -0.5, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.vertex_buffer = (
            BufferBuilder()
            .from_data(vertex_data)
            .with_usage(BufferUsage.VERTEX)
            .build()
        )

        self.pipeline = (
            GraphicPipelineBuilder()
            .with_shader("shader.wgsl")
            .with_vertex_buffer()
            .with_attribute(VertexFormat.float32x2)
            .with_attribute(VertexFormat.float32x3)
            .build()
        )

    def render(self, screen: GPUTexture):
        command_buffer_builder = CommandBufferBuilder()

        render_pass = command_buffer_builder.begin_render_pass(screen).build()
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(3)
        render_pass.end()

        command_buffer_builder.submit()


Window((800, 600)).run(MyApp())
