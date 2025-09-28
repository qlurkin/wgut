from wgpu.gui.glfw import WgpuCanvas, run
import wgpu
from wgpu import GPUCanvasContext

from .core import get_device
import time


class Window:
    def __init__(self, size: tuple[int, int] = (800, 600), max_fps: int = 60):
        device = get_device()

        self.canvas = WgpuCanvas(title="WGPU Window", size=size, max_fps=max_fps)
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

    def set_title(self, title: str):
        self.canvas.set_title(title)

    def get_canvas(self) -> WgpuCanvas:
        return self.canvas

    def run(self):
        self.setup()
        prev_time = None

        def main_loop():
            nonlocal prev_time
            current_time = time.perf_counter()
            # canvas_texture = self.present_context.get_current_texture()
            if prev_time is None:
                frame_time = 0
            else:
                frame_time = current_time - prev_time
            prev_time = current_time
            self.update(frame_time)
            self.render()
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
