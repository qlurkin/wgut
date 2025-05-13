import numpy as np
from numpy.typing import NDArray


class BasicColorMaterial:
    def __init__(self, color: tuple[float, float, float]):
        self.__color = color
        self.__data = np.array(color + (1.0,), dtype=np.float32)

    @staticmethod
    def get_fragment() -> str:
        return """
            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                var lightDir = vec3<f32>(1.0, 1.0, -1.0);
                var shading = clamp(dot(normalize(in.normal), normalize(lightDir)), 0.0, 1.0);
                var color = vec3<f32>(1.0, 0.0, 0.0);
                return vec4<f32>(shading * color, 1.0);
            }
        """

    def get_data(self) -> NDArray:
        return self.__data

    @staticmethod
    def get_data_size() -> int:
        return 16

    def __eq__(self, other):
        return self.__color == other.__color

    def __hash__(self):
        return hash(self.__color)
