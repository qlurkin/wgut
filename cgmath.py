import numpy as np
import numpy.typing as ntp
import math


def vec(x: ntp.ArrayLike) -> ntp.NDArray:
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("Vec must have 1 dimention")
    return x


def vec3(x: ntp.ArrayLike) -> ntp.NDArray:
    x = vec(x)
    if x.size != 3:
        raise ValueError("Vec3 must be of size 3")
    return x


def normalize(v: ntp.ArrayLike) -> ntp.NDArray:
    v = vec(v)
    length = np.linalg.norm(v)
    return v / length


def look_at(
    eye: ntp.ArrayLike, target: ntp.ArrayLike, up: ntp.ArrayLike
) -> ntp.NDArray:
    """
    Convert world coordinates to camera coordinates where the camera point in the direction of -z axis and up is the y axis.
    """
    eye = vec3(eye)
    target = vec3(target)
    up = vec3(up)
    dir = target - eye
    f = normalize(dir)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    # fmt: off
    # Warning: Matrix in column major order
    return np.array([[    s[0],     u[0],   -f[0], 0],
                     [    s[1],     u[1],   -f[1], 0],
                     [    s[2],     u[2],   -f[2], 0],
                     [-eye @ s, -eye @ u, eye @ f, 1]], dtype=np.float32)
    # fmt: on


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> ntp.NDArray:
    """
    Convert camera coordinates to wgpu clip coordinates where x and y goes from -1.0 to 1.0 and z goes from 0.0 to 1.0.
    """
    angle = fovy_deg * math.pi / 180
    yspan = near * math.tan(angle)
    xspan = yspan * aspect

    c0r0 = 2 * near / xspan
    c1r1 = 2 * near / yspan
    c2r2 = -(far + near) / (far - near) / 2 - 0.5
    c3r2 = -far * near / (far - near)

    # fmt: off
    # Warning: Matrix in column major order
    return np.array([[c0r0,    0,    0,    0],
                     [   0, c1r1,    0,    0],
                     [   0,    0, c2r2,   -1],
                     [   0,    0, c3r2,    0]], dtype=np.float32)
    # fmt: on
