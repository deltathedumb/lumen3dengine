import math

from ._vector import Vector3


class Matrix4:
    """Row-major 4x4 matrix backed by a flat list[float] of 16 entries."""

    m: list[float]

    def __init__(self, m: list[float]):
        self.m = m

    @staticmethod
    def identity():
        return Matrix4([
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])

    @staticmethod
    def translation(x: float, y: float, z: float):
        return Matrix4([
            1.0,
            0.0,
            0.0,
            x,
            0.0,
            1.0,
            0.0,
            y,
            0.0,
            0.0,
            1.0,
            z,
            0.0,
            0.0,
            0.0,
            1.0,
        ])

    @staticmethod
    def scale(x: float, y: float, z: float):
        return Matrix4([
            x,
            0.0,
            0.0,
            0.0,
            0.0,
            y,
            0.0,
            0.0,
            0.0,
            0.0,
            z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])

    @staticmethod
    def rotation_x(theta: float):
        c: float = math.cos(theta)
        s: float = math.sin(theta)
        return Matrix4([
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            c,
            -s,
            0.0,
            0.0,
            s,
            c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])

    @staticmethod
    def rotation_y(theta: float):
        c: float = math.cos(theta)
        s: float = math.sin(theta)
        return Matrix4([
            c,
            0.0,
            s,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            -s,
            0.0,
            c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])

    @staticmethod
    def rotation_z(theta: float):
        c: float = math.cos(theta)
        s: float = math.sin(theta)
        return Matrix4([
            c,
            -s,
            0.0,
            0.0,
            s,
            c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])

    @staticmethod
    def perspective(fov_deg: float, aspect: float, near: float, far: float):
        fov_rad: float = fov_deg * math.pi / 180.0
        # Binding the math.tan() call to its own variable before dividing is
        # required here -- `1.0 / math.tan(...)` used directly crashes at
        # runtime (confirmed compiler codegen bug: using a function call's
        # result inline inside an arithmetic expression, rather than from a
        # variable, corrupts state).
        half_fov_tan: float = math.tan(fov_rad / 2.0)
        f: float = 1.0 / half_fov_tan
        return Matrix4([
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            (far + near) / (near - far),
            (2.0 * far * near) / (near - far),
            0.0,
            0.0,
            -1.0,
            0.0,
        ])

    @staticmethod
    def look_at(eye: Vector3, target: Vector3, up: Vector3):
        forward: Vector3 = eye - target
        z: Vector3 = forward.normalized()
        x_raw: Vector3 = up.cross(z)
        x: Vector3 = x_raw.normalized()
        y: Vector3 = z.cross(x)
        # Same inline-call-result bug as perspective()'s math.tan() division:
        # bind each .dot() call's result to a variable before negating it.
        x_dot_eye: float = x.dot(eye)
        y_dot_eye: float = y.dot(eye)
        z_dot_eye: float = z.dot(eye)
        return Matrix4([
            x.x,
            x.y,
            x.z,
            -x_dot_eye,
            y.x,
            y.y,
            y.z,
            -y_dot_eye,
            z.x,
            z.y,
            z.z,
            -z_dot_eye,
            0.0,
            0.0,
            0.0,
            1.0,
        ])

    def multiply(self, other: Matrix4):
        """Returns self * other (other is applied first when transforming a point)."""
        result: list[float] = []
        i: int = 0
        while i < 4:
            j: int = 0
            while j < 4:
                s: float = 0.0
                k: int = 0
                while k < 4:
                    s = s + self.m[i * 4 + k] * other.m[k * 4 + j]
                    k = k + 1
                result.append(s)
                j = j + 1
            i = i + 1
        return Matrix4(result)

    def transform_point(self, v: Vector3) -> list[float]:
        """Returns the homogeneous [x, y, z, w] result of M * (v.x, v.y, v.z, 1).

        list[float], not tuple[float, ...]: reading back a returned
        tuple[float, ...] by index gives the raw IEEE-754 bit pattern
        reinterpreted as int64 instead of the float value (confirmed
        compiler codegen bug). list[float] round-trips correctly.
        """
        x: float = self.m[0] * v.x + self.m[1] * v.y + self.m[2] * v.z + self.m[3]
        y: float = self.m[4] * v.x + self.m[5] * v.y + self.m[6] * v.z + self.m[7]
        z: float = self.m[8] * v.x + self.m[9] * v.y + self.m[10] * v.z + self.m[11]
        w: float = self.m[12] * v.x + self.m[13] * v.y + self.m[14] * v.z + self.m[15]
        return [x, y, z, w]

    def transform_direction(self, v: Vector3) -> list[float]:
        """Like transform_point(), but for a direction (e.g. a surface
        normal) rather than a position -- applies only the 3x3 linear part,
        skipping translation (the rightmost column), since a direction has
        no position to translate.

        Correct for any model matrix built purely from rotation (and
        uniform scale), which is all Renderer3D currently constructs; a
        non-uniform scale would need the inverse-transpose instead.
        """
        x: float = self.m[0] * v.x + self.m[1] * v.y + self.m[2] * v.z
        y: float = self.m[4] * v.x + self.m[5] * v.y + self.m[6] * v.z
        z: float = self.m[8] * v.x + self.m[9] * v.y + self.m[10] * v.z
        return [x, y, z]
