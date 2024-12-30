from wgpu.gui.glfw import WgpuCanvas, run
import wgpu
from gpu import get_device
import time
from typing import Self


class Window:
    def __init__(self):
        self.size = (800, 600)
        self.title = "WGPU Window"
        self.max_fps = 60

    def with_size(self, size: tuple[int, int]) -> Self:
        self.size = size
        return self

    def with_max_fps(self, max_fps: int) -> Self:
        self.max_fps = max_fps
        return self

    def with_title(self, title: str) -> Self:
        self.title = title
        return self

    def run(self):
        self.setup(self.size)
        device = get_device()
        canvas = WgpuCanvas(title=self.title, size=self.size, max_fps=self.max_fps)
        present_context = canvas.get_context("wgpu")
        format = wgpu.TextureFormat.bgra8unorm
        present_context.configure(device=device, format=format)
        prev_time = None

        def loop():
            nonlocal prev_time
            current_time = time.perf_counter()
            canvas_texture = present_context.get_current_texture()
            if prev_time is None:
                frame_time = 0
            else:
                frame_time = current_time - prev_time
            prev_time = current_time
            self.update(frame_time)
            self.render(canvas_texture)
            canvas.request_draw()

        canvas.request_draw(loop)
        run()

    def setup(self, size: tuple[int, int]):
        pass

    def update(self, delta_time: float):
        pass

    def render(self, screen: wgpu.GPUTexture):
        pass
