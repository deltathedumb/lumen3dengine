"""AABB -- axis-aligned bounding box collision, the simplest real physics
primitive (no rotated/oriented boxes, no mesh-accurate convex hulls --
just enough for "did these two things touch", which is what Instance's
`touched` signal needs to mean something).

World-space AABBs are recomputed from each Instance's local-space mesh
bounds (cached on Mesh the first time it's needed) transformed by the
Instance's current world matrix -- correct under rotation (the box grows
to contain the rotated shape, the standard AABB-under-rotation behavior
every engine's broad-phase uses) rather than trying to track a rotated
box, which would need real OBB (oriented bounding box) math.
"""
from __future__ import annotations

from pugtk._vector import Vector3
from pugtk._matrix import Matrix4
from pugtk._mesh import Mesh


class AABB:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def __init__(self, min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def overlaps(self, other: "AABB") -> int:
        if self.max_x < other.min_x or other.max_x < self.min_x:
            return 0
        if self.max_y < other.min_y or other.max_y < self.min_y:
            return 0
        if self.max_z < other.min_z or other.max_z < self.min_z:
            return 0
        return 1


def _local_bounds(mesh: Mesh) -> AABB:
    """Local-space (untransformed) AABB of every vertex in `mesh`."""
    first: Vector3 = mesh.vertices[0]
    min_x: float = first.x
    min_y: float = first.y
    min_z: float = first.z
    max_x: float = first.x
    max_y: float = first.y
    max_z: float = first.z
    i: int = 1
    while i < len(mesh.vertices):
        v: Vector3 = mesh.vertices[i]
        if v.x < min_x:
            min_x = v.x
        if v.y < min_y:
            min_y = v.y
        if v.z < min_z:
            min_z = v.z
        if v.x > max_x:
            max_x = v.x
        if v.y > max_y:
            max_y = v.y
        if v.z > max_z:
            max_z = v.z
        i = i + 1
    return AABB(min_x, min_y, min_z, max_x, max_y, max_z)


def world_aabb(mesh: Mesh, world_matrix: Matrix4) -> AABB:
    """Transforms `mesh`'s local AABB's 8 corners by `world_matrix` and
    re-encloses them -- correct (if slightly loose) under any rotation,
    since the result is still the tightest AXIS-ALIGNED box, not a
    rotated one."""
    local: AABB = _local_bounds(mesh)
    corners: list[Vector3] = [
        Vector3(local.min_x, local.min_y, local.min_z),
        Vector3(local.max_x, local.min_y, local.min_z),
        Vector3(local.min_x, local.max_y, local.min_z),
        Vector3(local.max_x, local.max_y, local.min_z),
        Vector3(local.min_x, local.min_y, local.max_z),
        Vector3(local.max_x, local.min_y, local.max_z),
        Vector3(local.min_x, local.max_y, local.max_z),
        Vector3(local.max_x, local.max_y, local.max_z),
    ]
    first: list[float] = world_matrix.transform_point(corners[0])
    min_x: float = first[0]
    min_y: float = first[1]
    min_z: float = first[2]
    max_x: float = first[0]
    max_y: float = first[1]
    max_z: float = first[2]
    i: int = 1
    while i < len(corners):
        p: list[float] = world_matrix.transform_point(corners[i])
        if p[0] < min_x:
            min_x = p[0]
        if p[1] < min_y:
            min_y = p[1]
        if p[2] < min_z:
            min_z = p[2]
        if p[0] > max_x:
            max_x = p[0]
        if p[1] > max_y:
            max_y = p[1]
        if p[2] > max_z:
            max_z = p[2]
        i = i + 1
    return AABB(min_x, min_y, min_z, max_x, max_y, max_z)
