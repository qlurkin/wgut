from typing import Callable
from numpy.typing import ArrayLike, NDArray
import numpy as np
from wgpu import BufferUsage, GPUBuffer

from wgut.builders.bufferbuilder import BufferBuilder
from wgut.core import read_buffer
from wgut.scene.ecs import ECS


class Particles:
    def __init__(
        self,
        translations: ArrayLike,
        callback: Callable[[ECS, GPUBuffer, float], None],
    ):
        self.__translations = (
            BufferBuilder()
            .from_data(np.array(translations))
            .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_SRC)
            .build()
        )
        self.__callback = callback

    def update(self, ecs: ECS, delta_time: float):
        self.__callback(ecs, self.__translations, delta_time)

    def get_translations(self) -> NDArray:
        mem = read_buffer(self.__translations)
        # translations = array.as_reference(mem).reinterpret_cast(vec4)  # type: ignore
        translations = np.frombuffer(mem.cast("f"), dtype=np.float32)
        translations = translations.reshape((translations.size // 4, 4))
        return translations
