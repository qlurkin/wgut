from wgpu import GPUTexture
from wgut import Window, load_file
import numpy as np
import wgut.cgmath as cm
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.renderer import Renderer


class MyApp(Window):
    def on_resize(self):
        canvas = self.get_canvas()

        width, height = canvas.get_logical_size()

        view_matrix = cm.look_at([10, 8, 9], [0, 0, 0], [0, 1, 0])
        proj_matrix = cm.perspective(45, width / height, 0.1, 100)

        # Must send transpose version of matrices, because GPU expect matrices in column major order
        camera_data = np.array([view_matrix.T, proj_matrix.T])

        self.renderer.set_binding_array(0, 0, camera_data)

    def setup(self):
        self.set_title("Hello Scene")

        self.mesh = icosphere(2)

        self.renderer = Renderer(load_file("scene.wgsl"), 1000, 10000)

        self.on_resize()

    def render(self, screen: GPUTexture):
        self.renderer.begin_frame(screen)
        self.renderer.add_mesh(self.mesh)
        self.renderer.add_mesh(self.mesh)
        self.renderer.add_mesh(self.mesh)
        self.renderer.end_frame()
        # print(self.renderer.get_frame_stat())

    def process_event(self, event):
        if event["event_type"] == "resize":
            self.on_resize()


MyApp().run()
