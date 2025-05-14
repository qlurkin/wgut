from wgpu import GPUTexture, VertexFormat
from wgut import TextureBuilder, AutoRenderPipeline, Window, load_file
from wgut.scene.primitives.icosphere import icosphere_with_uv
import numpy as np
import wgut.cgmath as cm


class MyApp(Window):
    def on_resize(self):
        canvas = self.get_canvas()
        self.pipeline.create_depth_texture(canvas.get_physical_size())

        width, height = canvas.get_logical_size()

        view_matrix = cm.look_at([0, 2, -4], [0, 0, 0], [0, 1, 0])
        proj_matrix = cm.perspective(45, width / height, 0.1, 100)

        # Must send transpose version of matrices, because GPU expect matrices in column major order
        camera_data = np.array([view_matrix.T, proj_matrix.T])

        self.pipeline.set_binding_array(0, 0, camera_data)

    def setup(self):
        self.set_title("Textured Icosphere")
        vertex_data_positions, vertex_data_uvs, indices = icosphere_with_uv(3)

        vertex_data = np.hstack([vertex_data_positions, vertex_data_uvs])

        self.pipeline = AutoRenderPipeline(load_file("./textured_icosphere.wgsl"))
        self.pipeline.add_simple_vertex_descriptor(
            VertexFormat.float32x3, VertexFormat.float32x2
        )

        self.pipeline.set_vertex_array(0, vertex_data)
        self.pipeline.set_index_array(indices)
        self.indices_count = len(indices)

        diffuse_texture = TextureBuilder().from_file("wood.jpg")

        self.pipeline.set_binding_texture(1, 0, diffuse_texture)
        self.pipeline.set_binding_sampler(1, 1)

        self.on_resize()

    def render(self, screen: GPUTexture):
        self.pipeline.render(screen, self.indices_count)

    def process_event(self, event):
        if event["event_type"] == "resize":
            self.on_resize()


MyApp().run()
