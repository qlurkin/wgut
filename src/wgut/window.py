from wgpu.gui.glfw import WgpuCanvas, run
import wgpu
from wgpu import GPUCanvasContext, GPUTexture

from .core import get_device
import time


class Window:
    def __init__(self, canvas: WgpuCanvas):
        device = get_device()

        self.last_render_time = 0.0
        self.last_update_time = 0.0
        self.last_frame_time = 0.0

        self.canvas = canvas
        self.present_context: GPUCanvasContext = self.canvas.get_context("wgpu")  # type: ignore
        self.texture_format: wgpu.TextureFormat = (
            self.present_context.get_preferred_format(device.adapter)
        )  # type: ignore

        self.present_context.configure(device=device, format=self.texture_format)

        def event_handler(event):
            self.process_event(event)

        self.canvas.add_event_handler(event_handler, "*")

    def get_texture_format(self) -> wgpu.TextureFormat:
        return self.texture_format

    def get_aspect(self) -> float:
        size = self.get_canvas().get_physical_size()
        return size[0] / size[1]

    def set_title(self, title: str):
        self.canvas.set_title(title)

    def get_canvas(self) -> WgpuCanvas:
        return self.canvas

    def get_current_texture(self) -> GPUTexture:
        return self.present_context.get_current_texture()

    def run(self):
        self.setup()
        prev_time = None

        def main_loop():
            nonlocal prev_time
            current_time = time.perf_counter()
            if prev_time is not None:
                self.last_frame_time = current_time - prev_time
            self.update(self.last_frame_time)
            mid = time.perf_counter()
            self.last_update_time = mid - current_time
            self.render()
            self.last_render_time = time.perf_counter() - mid
            prev_time = current_time
            self.canvas.request_draw()  # pyright: ignore

        self.canvas.request_draw(main_loop)
        run()

    def setup(self):
        pass

    def update(self, delta_time: float):
        pass

    def render(self):
        pass

    def process_event(self, event):
        pass
