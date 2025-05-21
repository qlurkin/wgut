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

    def process_event(self, event):
        self.__ecs.dispatch("window_event", event)


def window_system(ecs: ECS):
    app = WindowSystemApp(ecs)
    ecs.spawn([app])
    app.run()
