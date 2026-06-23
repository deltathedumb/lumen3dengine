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
                if aabb_a.overlaps(aabb_b):
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
