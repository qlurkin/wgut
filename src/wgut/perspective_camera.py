from numpy.typing import ArrayLike
from pygfx import PerspectiveCamera as PgfxPerspectiveCamera


class PerspectiveCamera(PgfxPerspectiveCamera):
    def __init__(
        self,
        fov: float | int = 50,
        aspect: float = 1.0,
        *,
        width: float | int | None = None,
        height: float | int | None = None,
        zoom: float = 1.0,
        maintain_aspect=True,
        depth: float | None = None,
        depth_range: tuple[float, float] | None = None,
    ):
        super().__init__(
            fov=fov,  # type: ignore
            aspect=aspect,  # type: ignore
            width=width,
            height=height,
            zoom=zoom,  # type: ignore
            maintain_aspect=maintain_aspect,
            depth=depth,
            depth_range=depth_range,
        )

    def look_at(self, target: ArrayLike):  # type: ignore
        super().look_at(target=target)  # type: ignore
