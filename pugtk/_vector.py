import math


class Vector2:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __add__(self, other: Vector2):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: int):
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: int):
        return Vector2(self.x // scalar, self.y // scalar)


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Vector3):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self):
        n: float = self.length()
        if n == 0.0:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / n, self.y / n, self.z / n)

    def normalize_in_place(self):
        """Like normalized(), but mutates self instead of allocating a new
        Vector3 -- for hot per-pixel loops (e.g. re-normalizing an
        interpolated Gouraud normal) that reuse one Vector3 object rather
        than allocating one per pixel."""
        n: float = self.length()
        if n == 0.0:
            return
        self.x = self.x / n
        self.y = self.y / n
        self.z = self.z / n
