import pygame
from pygame import Surface
from gpu import Texture

pygame.init()
pygame.font.init()

FONT = pygame.Font(None, 30)


def surface_from_texture(texture: Texture):
    surface = pygame.image.frombuffer(
        texture.get_memoryview().tobytes(), texture.size, "RGBA"
    )
    return surface


class App:
    def setup(self, size: tuple[int, int]):
        pass

    def update(self, delta_time: float):
        pass

    def render(self, screen: Surface):
        pass


class Window:
    def __init__(self, size: tuple[int, int]):
        self.screen = pygame.display.set_mode(size)

    def run(self, app: App):
        clock = pygame.time.Clock()
        app.setup(self.screen.size)
        while not pygame.event.peek(pygame.QUIT):
            frame_time = clock.tick(60)
            app.update(frame_time)
            app.render(self.screen)
            self.screen.blit(
                FONT.render(f"Frame Time: {frame_time:.5f} ms", False, (255, 255, 255))
            )
            pygame.display.flip()
