import numpy as np
from numpy.typing import NDArray


class BasicColorMaterial:
    def __init__(self, color: tuple[float, float, float]):
        self.__color = color
        self.__data = np.array(color + (1.0,), dtype=np.float32)

    @staticmethod
    def get_fragment() -> str:
        return """
            struct Material {
                color: vec4<f32>,
            }

            @group(1) @binding(0) var<storage, read> materials: array<Material>;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                var color = materials[u32(in.mat_id)].color;
                return color;
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
