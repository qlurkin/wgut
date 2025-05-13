from typing import Self
import numpy as np
from numpy.typing import NDArray


class VertexColorMaterial:
    @staticmethod
    def get_fragment() -> str:
        return """
            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return in.color;
            }
        """

    def get_data(self) -> NDArray:
        return np.array([])

    @staticmethod
    def get_data_size() -> int:
        return 0

    def __eq__(self, other):
        return isinstance(other, Self)

    def __hash__(self):
        return hash("VertexColorMaterial")
