from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx

from wgut.core import create_texture

canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


image = gfx.Image(
    gfx.Geometry(grid=create_texture("./textures/texel_checker.png")),
    gfx.ImageBasicMaterial(clim=(0, 255), pick_write=True),
)
scene.add(image)

ball = gfx.Mesh(
    gfx.sphere_geometry(1),
    gfx.MeshBasicMaterial(color=(1.0, 0.0, 0.0), pick_write=True),
)
scene.add(ball)


camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.position = (2, 2, 2)
camera.look_at((0, 0, 0))


def event_handler(event):
    print(event)


image.add_event_handler(event_handler, "click")
ball.add_event_handler(event_handler, "pointer_move")


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
