from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import numpy as np

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

# camera = gfx.OrthographicCamera(512, 512)
# camera.local.position = (256, 256, 0)
# camera.local.scale_y = -1

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.position = (2, 2, 2)
camera.look_at((0, 0, 0))


def event_handler(event):
    print(event)
    # print(
    #     f"Canvas click coordinates: {event.x, event.y}\n"
    #     f"Click position in coordinate system of image, i.e. data coordinates of click event: {event.pick_info['index']}\n"
    #     f"Other `pick_info`: {event.pick_info}"
    # )
    #
    # global previous_selection
    #
    # if previous_selection is not None:
    #     # reset colors to blue
    #     blues = np.vstack([[0.0, 0.0, 1.0, 1.0]] * 3).astype(np.float32)
    #     points.geometry.colors.data[previous_selection] = blues
    #
    #     for idx in previous_selection.tolist():
    #         points.geometry.colors.update_range(idx, size=1)
    #
    # # set the color of the 3 closest points to red
    # positions = points.geometry.positions.data
    # event_position = np.array([*event.pick_info["index"], 0])
    # closest = np.linalg.norm(positions - event_position, axis=1).argsort()
    # points.geometry.colors.data[closest[:3]] = np.array([1.0, 0.0, 0.0, 1.0])
    # for idx in closest[:3].tolist():
    #     # only mark the changed points for synchronization to the GPU
    #     points.geometry.colors.update_range(idx, size=1)
    #
    # previous_selection = closest[:3]
    #
    # renderer.request_draw()


image.add_event_handler(event_handler, "click")
ball.add_event_handler(event_handler, "pointer_move")


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
