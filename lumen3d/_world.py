"""World -- owns the Instance tree and renders it through pugtk's
GLRenderer3D each frame.

This is the bridge between lumen3dengine's scripting layer (Instance,
Signal) and pugtk's graphics layer (GLRenderer3D, SceneObject, Mesh):
World walks the Instance tree, computes each Instance's world-space
transform, and builds the flat SceneObject list GLRenderer3D.render_scene()
actually consumes -- pugtk itself has no notion of a persistent scene
graph or game-object identity, by design (see pugtk's own docstrings).
"""
from __future__ import annotations

from pugtk._scene import SceneObject
from pugtk._renderer3d_gl import GLRenderer3D
from pugtk._vector import Vector3

from ._instance import Instance
from ._aabb import world_aabb, AABB


class World:
    renderer: GLRenderer3D
    roots: list
    collision_enabled: int

    def __init__(self, renderer: GLRenderer3D) -> None:
        self.renderer = renderer
        self.roots = []
        self.collision_enabled = 1

    def add(self, instance: Instance) -> None:
        self.roots.append(instance)

    def _collect_all(self, instance: Instance, out: list) -> None:
        out.append(instance)
        i: int = 0
        while i < len(instance.children):
            self._collect_all(instance.children[i], out)
            i = i + 1

    def _resolve_collision(self, a: Instance, b: Instance, mtv: list[float]) -> None:
        """Push a and b apart by mtv, then cancel (or bounce via restitution)
        each object's velocity along the collision normal.

        restitution=0.0 (default): perfectly inelastic (no bounce).
        restitution=1.0: perfectly elastic (full bounce).
        Combined restitution is the average of both objects' values.

        Anchored instances are never moved. Non-anchored side absorbs the
        full push; two mobile objects split 50/50."""
        dx: float = mtv[0]
        dy: float = mtv[1]
        dz: float = mtv[2]

        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return

        e: float = (a.restitution + b.restitution) * 0.5
        bounce: float = -(1.0 + e)

        a_free: int = 1 if a.anchored == 0 else 0
        b_free: int = 1 if b.anchored == 0 else 0

        if a_free == 1 and b_free == 0:
            a.position = Vector3(
                a._position.x + dx,
                a._position.y + dy,
                a._position.z + dz,
            )
            if dx != 0.0:
                a.velocity = Vector3(a.velocity.x * bounce, a.velocity.y, a.velocity.z)
            if dy != 0.0:
                a.velocity = Vector3(a.velocity.x, a.velocity.y * bounce, a.velocity.z)
            if dz != 0.0:
                a.velocity = Vector3(a.velocity.x, a.velocity.y, a.velocity.z * bounce)
        elif a_free == 0 and b_free == 1:
            b.position = Vector3(
                b._position.x - dx,
                b._position.y - dy,
                b._position.z - dz,
            )
            if dx != 0.0:
                b.velocity = Vector3(b.velocity.x * bounce, b.velocity.y, b.velocity.z)
            if dy != 0.0:
                b.velocity = Vector3(b.velocity.x, b.velocity.y * bounce, b.velocity.z)
            if dz != 0.0:
                b.velocity = Vector3(b.velocity.x, b.velocity.y, b.velocity.z * bounce)
        elif a_free == 1 and b_free == 1:
            half_dx: float = dx * 0.5
            half_dy: float = dy * 0.5
            half_dz: float = dz * 0.5
            a.position = Vector3(
                a._position.x + half_dx,
                a._position.y + half_dy,
                a._position.z + half_dz,
            )
            b.position = Vector3(
                b._position.x - half_dx,
                b._position.y - half_dy,
                b._position.z - half_dz,
            )
            if dx != 0.0:
                a.velocity = Vector3(a.velocity.x * bounce, a.velocity.y, a.velocity.z)
                b.velocity = Vector3(b.velocity.x * bounce, b.velocity.y, b.velocity.z)
            if dy != 0.0:
                a.velocity = Vector3(a.velocity.x, a.velocity.y * bounce, a.velocity.z)
                b.velocity = Vector3(b.velocity.x, b.velocity.y * bounce, b.velocity.z)
            if dz != 0.0:
                a.velocity = Vector3(a.velocity.x, a.velocity.y, a.velocity.z * bounce)
                b.velocity = Vector3(b.velocity.x, b.velocity.y, b.velocity.z * bounce)

    def _check_collisions(self, instances: list) -> None:
        n: int = len(instances)
        i: int = 0
        while i < n:
            a: Instance = instances[i]
            if a.mesh is None:
                i = i + 1
                continue
            aabb_a: AABB = world_aabb(a.mesh, a.world_matrix())
            j: int = i + 1
            while j < n:
                b: Instance = instances[j]
                if b.mesh is None:
                    j = j + 1
                    continue
                aabb_b: AABB = world_aabb(b.mesh, b.world_matrix())
                mtv: list[float] = aabb_a.penetration(aabb_b)
                if mtv[0] != 0.0 or mtv[1] != 0.0 or mtv[2] != 0.0:
                    self._resolve_collision(a, b, mtv)
                    a.touched(j)
                    b.touched(i)
                j = j + 1
            i = i + 1

    def step(self) -> None:
        """Runs collision detection for the current frame (called by
        GameLoop before render). Fires Instance.touched(other_index)
        on every overlapping pair -- the argument is the flat-list
        index of the other Instance, which scripts can use to look
        up the collider in their own references."""
        if self.collision_enabled == 0:
            return
        all_instances: list = []
        i: int = 0
        while i < len(self.roots):
            self._collect_all(self.roots[i], all_instances)
            i = i + 1
        self._check_collisions(all_instances)

    def render(self) -> None:
        scene_objects: list = []
        i: int = 0
        while i < len(self.roots):
            self._collect_render(self.roots[i], scene_objects)
            i = i + 1
        self.renderer.render_scene(scene_objects)

    def _collect_render(self, instance: Instance, out: list) -> None:
        if instance.mesh is not None:
            scene_obj = SceneObject(instance.mesh, instance.world_matrix(), instance.colors, instance.texture)
            out.append(scene_obj)
        i: int = 0
        while i < len(instance.children):
            self._collect_render(instance.children[i], out)
            i = i + 1
