from typing import Protocol
import numpy.typing as npt


class Material(Protocol):
    @staticmethod
    def get_fragment() -> str: ...
    @staticmethod
    def get_data_size() -> int: ...
    def get_data(self) -> npt.NDArray: ...
