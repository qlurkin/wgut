from wgpu import VertexFormat
from gpu import Texture, GraphicPipelineBuilder, Buffer, RenderPass
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

        self.vertex_buffer = Buffer(vertex_data)

        self.pipeline = (
            GraphicPipelineBuilder()
            .with_shader("shader.wgsl")
            .with_vertex_buffer()
            .with_attribute(VertexFormat.float32x2)
            .with_attribute(VertexFormat.float32x3)
            .build()
        )

    def render(self, screen: Texture):
        RenderPass(screen).with_pipeline(self.pipeline).with_vertex_buffer(
            self.vertex_buffer
        ).draw(3).submit()


Window((800, 600)).run(MyApp())
