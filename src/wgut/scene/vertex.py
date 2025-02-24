import numpy.typing as npt
import numpy as np
import wgpu


def vertex(
    position: npt.NDArray,
    color: npt.NDArray | None = None,
    tex_coord: npt.NDArray | None = None,
    normal: npt.NDArray | None = None,
    tangent: npt.NDArray | None = None,
    bitangent: npt.NDArray | None = None,
):
    if color is None:
        color = np.array([1.0, 1.0, 1.0])
    if tex_coord is None:
        tex_coord = np.array([0.0, 0.0])
    if normal is None:
        normal = np.array([0.0, 0.0, 0.0])
    if tangent is None:
        tangent = np.array([0.0, 0.0, 0.0])
    if bitangent is None:
        bitangent = np.array([0.0, 0.0, 0.0])
    return np.array([position, color, tex_coord, normal, tangent, bitangent]).flatten()


def get_vertex_buffer_descriptor():
    return {
        "array_stride": 4 * (3 + 3 + 2 + 3 + 3 + 3),
        "step_mode": wgpu.VertexStepMode.vertex,
        "attributes": [
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": 0,
                "shader_location": 0,
            },
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": 3 * 4,
                "shader_location": 1,
            },
            {
                "format": wgpu.VertexFormat.float32x2,
                "offset": (3 + 3) * 4,
                "shader_location": 2,
            },
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": (3 + 3 + 2) * 4,
                "shader_location": 3,
            },
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": (3 + 3 + 2 + 3) * 4,
                "shader_location": 4,
            },
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": (3 + 3 + 2 + 3 + 3) * 4,
                "shader_location": 5,
            },
        ],
    }
