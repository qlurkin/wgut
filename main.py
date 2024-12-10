from gpu import Texture, GraphicPipeline
from window import Window, surface_from_texture


class App:
    def setup(self):
        self.texture = Texture((800, 600))
        self.pipeline = GraphicPipeline("shader.wgsl")

    def update(self, delta_time):
        pass

    def render(self, screen):
        self.pipeline.render(self.texture)
        surface = surface_from_texture(self.texture)
        screen.blit(surface)


Window((800, 600)).run(App())
