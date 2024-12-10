import pygame
from gpu import Texture

pygame.init()
pygame.font.init()

FONT = pygame.Font(None, 30)


def surface_from_texture(texture: Texture):
    surface = pygame.image.frombuffer(
        texture.get_memoryview().tobytes(), texture.size, "RGBA"
    )
    return surface


class Window:
    def __init__(self, size):
        self.screen = pygame.display.set_mode(size)

    def run(self, app):
        clock = pygame.time.Clock()
        app.setup()
        while not pygame.event.peek(pygame.QUIT):
            frame_time = clock.tick(60)
            app.update(frame_time)
            app.render(self.screen)
            self.screen.blit(
                FONT.render(f"Frame Time: {frame_time:.5f} ms", False, (255, 255, 255))
            )
            pygame.display.flip()
