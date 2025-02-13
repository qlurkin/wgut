from wgpu import GPUTexture, VertexFormat
from wgut.auto_render_pipeline import AutoRenderPipeline
from wgut.builders import (
    TextureBuilder,
)
from wgut.window import Window
import numpy as np


class MyApp(Window):
    def setup(self):
        self.set_title("Hello Texture")
        vertex_data = np.array(
            [
                # x, y, u, v
                [0.0, 0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.0, 1.0],
                [0.5, -0.5, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.pipeline = AutoRenderPipeline("./texture.wgsl")
        self.pipeline.add_simple_vertex_descriptor(
            VertexFormat.float32x2, VertexFormat.float32x2
        )

        self.pipeline.set_vertex_array(0, vertex_data)

        diffuse_texture = TextureBuilder().from_file("wood.jpg")

        self.pipeline.set_binding_texture(0, 0, diffuse_texture)
        self.pipeline.set_binding_sampler(0, 1)

    def render(self, screen: GPUTexture):
        self.pipeline.render(screen, 3)


MyApp().run()
