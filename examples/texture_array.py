from wgpu import GPUTexture, TextureFormat, TextureUsage, VertexFormat
from wgut import TextureBuilder, AutoRenderPipeline, Window, load_file
import numpy as np

from wgut.core import load_image, write_texture


class MyApp(Window):
    def setup(self):
        self.set_title("Hello Texture")

        vertex_data = np.array(
            [
                # x, y, u, v, id
                [-0.5, 0.5, 0.5, 0.0, 0.0],
                [-1.0, -0.5, 0.0, 1.0, 0.0],
                [0.0, -0.5, 1.0, 1.0, 0.0],
                [0.5, 0.5, 0.5, 0.0, 1.0],
                [0.0, -0.5, 0.0, 1.0, 1.0],
                [1.0, -0.5, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.pipeline = AutoRenderPipeline(load_file("./texture_array.wgsl"))
        self.pipeline.add_simple_vertex_descriptor(
            VertexFormat.float32x2, VertexFormat.float32x2, VertexFormat.float32
        )

        self.pipeline.set_vertex_array(0, vertex_data)

        image0 = load_image("./wood.jpg")
        image1 = load_image("./cloth.jpg")

        texture = (
            TextureBuilder()
            .with_size((image0.width, image0.height, 2))
            .with_format(TextureFormat.rgba8unorm_srgb)
            .with_usage(TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING)
            .build()
        )

        write_texture(texture, image0)
        write_texture(texture, image1, 1)

        self.pipeline.set_binding_texture(0, 0, texture)
        self.pipeline.set_binding_sampler(0, 1)

    def render(self, screen: GPUTexture):
        self.pipeline.render(screen, 6)


MyApp().run()
