from gpu import Texture, GraphicPipeline
from window import Window, App, Surface, surface_from_texture


class MyApp(App):
    def setup(self, size: tuple[int, int]):
        self.texture = Texture(size)
        self.pipeline = GraphicPipeline("shader.wgsl")

    def render(self, screen: Surface):
        self.pipeline.render(self.texture)
        surface = surface_from_texture(self.texture)
        screen.blit(surface)


Window((800, 600)).run(MyApp())
