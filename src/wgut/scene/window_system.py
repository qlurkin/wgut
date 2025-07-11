from wgpu import GPUTexture
from wgut.scene.ecs import ECS
from wgut import Window


class WindowSystemApp(Window):
    def __init__(self, ecs: ECS):
        super().__init__()
        self.__ecs = ecs

    def setup(self):
        self.__ecs.dispatch("setup")

    def update(self, delta_time: float):
        self.__ecs.dispatch("update", delta_time)

    def render(self, screen: GPUTexture):
        self.__ecs.dispatch("render", screen)
        self.__ecs.dispatch("after_render", screen)

    def process_event(self, event):
        self.__ecs.dispatch("window_event", event)

    def __str__(self):
        return "WindowSystemApp"


def window_system(ecs: ECS, title="WGUT Window"):
    app = WindowSystemApp(ecs)
    app.set_title(title)
    ecs.spawn([app])
    app.run()
