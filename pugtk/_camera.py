from ._vector import Vector3
from ._matrix import Matrix4


class Camera:
    position: Vector3
    target: Vector3
    up: Vector3

    def __init__(
        self,
        position: Vector3,
        target: Vector3,
        up: Vector3,
        fov_deg: float,
        aspect: float,
        near: float,
        far: float,
    ):
        self.position = position
        self.target = target
        self.up = up
        self.fov_deg = fov_deg
        self.aspect = aspect
        self.near = near
        self.far = far

    def view_matrix(self) -> Matrix4:
        return Matrix4.look_at(self.position, self.target, self.up)

    def projection_matrix(self) -> Matrix4:
        return Matrix4.perspective(self.fov_deg, self.aspect, self.near, self.far)
