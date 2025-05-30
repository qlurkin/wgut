from pyglm.glm import array
from wgpu import GPUTexture, VertexFormat
from wgut import TextureBuilder, AutoRenderPipeline, Window, load_file
from wgut.orbit_camera import OrbitCamera
from wgut.scene.primitives.icosphere import icosphere_with_uv
import numpy as np


class MyApp(Window):
    def setup(self):
        self.set_title("Textured Icosphere")
        vertex_data_positions, vertex_data_uvs, indices = icosphere_with_uv(3)

        vertex_data = np.hstack([vertex_data_positions, vertex_data_uvs])

        self.camera = OrbitCamera((0, 2, -4), (0, 0, 0), 45, 0.1, 100)

        self.pipeline = AutoRenderPipeline(load_file("./textured_icosphere.wgsl"))
        self.pipeline.with_depth_texture()
        self.pipeline.add_simple_vertex_descriptor(
            VertexFormat.float32x3, VertexFormat.float32x2
        )

        self.pipeline.set_vertex_array(0, vertex_data)
        self.pipeline.set_index_array(indices)
        self.indices_count = len(indices)

        diffuse_texture = TextureBuilder().from_file(
            "./textures/Wood_025_basecolor.jpg"
        )

        self.pipeline.set_binding_texture(1, 0, diffuse_texture)
        self.pipeline.set_binding_sampler(1, 1)

    def render(self, screen: GPUTexture):
        view_matrix, proj_matrix = self.camera.get_matrices(
            screen.width / screen.height
        )

        # Must send transpose version of matrices, because GPU expect matrices in column major order
        camera_data = array(view_matrix, proj_matrix)

        self.pipeline.set_binding_array(0, 0, camera_data.to_bytes())

        self.pipeline.render(screen, self.indices_count)

    def process_event(self, event):
        self.camera.process_event(event)


MyApp().run()
