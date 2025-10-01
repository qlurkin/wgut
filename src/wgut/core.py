import PIL.Image as img
import wgpu
import pygfx as gfx
import numpy.typing as npt
import numpy as np
from wgpu.gui.glfw import WgpuCanvas


_SHARED = None


def get_shared():
    global _SHARED
    if _SHARED is None:
        gfx.renderers.wgpu.select_power_preference("high-performance")
        _SHARED = gfx.renderers.wgpu.get_shared()
        assert _SHARED is not None
    return _SHARED


def get_adapter() -> wgpu.GPUAdapter:
    return get_shared().adapter


def print_adapter_info():
    adapter = get_adapter()

    def title(text):
        return "\n" + text + ":\n" + "-" * (len(text) + 1)

    print(title("INFO"))
    for key, value in adapter.info.items():
        print(f" - {key}: {value}")
    print(f" - is_software: {adapter.is_fallback_adapter}")
    print(title("LIMITS"))
    for key, value in adapter.limits.items():
        print(f" - {key}: {value}")
    print(title("FEATURES"))
    for item in adapter.features:
        print(f" - {item}")
    print()


def get_device() -> wgpu.GPUDevice:
    return get_shared().device


def read_buffer(buffer: wgpu.GPUBuffer) -> memoryview:
    return get_device().queue.read_buffer(buffer)


def read_pygfx_buffer(buffer: gfx.Buffer) -> memoryview:
    wgpu_buffer: wgpu.GPUBuffer = buffer._wgpu_object  # type: ignore
    return read_buffer(wgpu_buffer)


def write_buffer(buffer: wgpu.GPUBuffer, data: npt.NDArray | bytes, buffer_offset=0):
    return get_device().queue.write_buffer(
        buffer=buffer, data=memoryview(data), buffer_offset=buffer_offset
    )


def write_pygfx_buffer(
    buffer: wgpu.GPUBuffer, data: npt.NDArray | bytes, buffer_offset=0
):
    wgpu_buffer: wgpu.GPUBuffer = buffer._wgpu_object  # type: ignore
    write_buffer(wgpu_buffer, data, buffer_offset)


def write_texture(
    texture: wgpu.GPUTexture, image: img.Image | bytes | npt.NDArray, index=0
):
    size = texture.size[:2]
    if isinstance(image, img.Image):
        if image.size != size:
            print(f"WARNING: Resize Image from {image.size} to {size}")
            image = image.resize(size)
        data = np.asarray(image)
        if image.mode == "RGB":
            # Add 'A' to get RGBA
            data = np.dstack((data, np.full(data.shape[:-1], 255, dtype=np.uint8)))
    else:
        data = image

    if not isinstance(data, bytes):
        data = data.tobytes()

    get_device().queue.write_texture(
        {
            "texture": texture,
            "mip_level": 0,
            "origin": (0, 0, index),
        },
        memoryview(data),
        {
            "offset": 0,
            "bytes_per_row": len(data) // size[1],
        },
        size,
    )


def load_image(filename) -> img.Image:
    return img.open(filename)


def create_texture(filename: str) -> gfx.Texture:
    img = load_image(filename)
    data = np.asarray(img)
    return gfx.Texture(data, dim=2)


def load_file(filename):
    with open(filename) as file:
        content = file.read()

    return content


def submit_command(command_encoder: wgpu.GPUCommandEncoder):
    get_device().queue.submit([command_encoder.finish()])


def create_canvas(
    size: tuple[int, int] = (800, 600), title="WGUT Window", max_fps=30, vsync=True
) -> WgpuCanvas:
    get_shared()
    return WgpuCanvas(size=size, title=title, max_fps=max_fps, vsync=vsync)
