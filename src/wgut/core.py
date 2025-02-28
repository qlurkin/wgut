import wgpu
import numpy.typing as npt

_ADAPTER = None
_DEVICE = None


def get_adapter() -> wgpu.GPUAdapter:
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = wgpu.gpu.request_adapter_sync(power_preference="high-performance")  # type: ignore
    return _ADAPTER


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
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = get_adapter().request_device_sync()
    return _DEVICE


def read_buffer(buffer: wgpu.GPUBuffer) -> memoryview:
    return get_device().queue.read_buffer(buffer)


def write_buffer(buffer: wgpu.GPUBuffer, data: npt.NDArray, buffer_offset=0):
    return get_device().queue.write_buffer(
        buffer=buffer, data=data, buffer_offset=buffer_offset
    )


def load_file(filename):
    with open(filename) as file:
        content = file.read()

    return content
