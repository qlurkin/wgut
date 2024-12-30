from wgpu.gui.glfw import WgpuCanvas, run
import wgpu
from gpu import get_device
import time


class App:
    def setup(self, size: tuple[int, int]):
        pass

    def update(self, delta_time: float):
        pass

    def render(self, screen: wgpu.GPUTexture):
        pass


class Window:
    def __init__(
        self,
        size: tuple[int, int] = (800, 600),
        title: str = "WGPU Window",
        max_fps: int = 60,
    ):
        device = get_device()
        self.size = size
        self.canvas = WgpuCanvas(title=title, size=size, max_fps=max_fps)
        self.present_context = self.canvas.get_context("wgpu")
        self.format = wgpu.TextureFormat.bgra8unorm
        self.present_context.configure(device=device, format=self.format)
        self.prev_time = None

    def run(self, app: App):
        def loop():
            canvas_texture = self.present_context.get_current_texture()
            if self.prev_time is None:
                frame_time = 0
            else:
                frame_time = time.perf_counter() - self.prev_time
            app.update(frame_time)
            app.render(canvas_texture)
            self.canvas.request_draw()

        app.setup(self.size)

        self.canvas.request_draw(loop)
        run()
