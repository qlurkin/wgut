from typing import Callable
from pyglm.glm import array, vec4
from wgpu import BufferUsage, GPUBuffer

from wgut.builders.bufferbuilder import BufferBuilder
from wgut.core import read_buffer
from wgut.scene.ecs import ECS


class Particles:
    def __init__(
        self,
        translations: array[vec4],
        callback: Callable[[ECS, GPUBuffer, float], None],
    ):
        self.__translations = (
            BufferBuilder()
            .from_data(translations)
            .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_SRC)
            .build()
        )
        self.__callback = callback

    def update(self, ecs: ECS, delta_time: float):
        self.__callback(ecs, self.__translations, delta_time)

    def get_translations(self) -> array[vec4]:
        mem = read_buffer(self.__translations)
        translations = array.as_reference(mem).reinterpret_cast(vec4)  # type: ignore
        return translations
