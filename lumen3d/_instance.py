"""Instance -- a persistent, named, scriptable game object (the
lumen3dengine analogue of Roblox's Instance/Part): position/rotation/scale
as reactive properties (writing any of them fires `changed`), a parent/
children tree, and a mesh+color+texture payload pugtk's GLRenderer3D can
actually draw.

This is the engine layer, built ON TOP of pugtk (the graphics layer) --
pugtk has no notion of game-object identity/hierarchy/scripting; it only
knows how to render a flat list of (mesh, transform, colors, texture)
tuples each frame (pugtk.SceneObject). World (_world.py) is what walks
an Instance tree and produces that flat per-frame list pugtk actually
consumes.

Position/rotation/scale are plain `@property`/`@x.setter` pairs (not
real `__setattr__` interception -- asmpython doesn't support that; see
the project's TODO.md). This gives the same call-site ergonomics
(`part.position = Vector3(1, 2, 3)` transparently fires `changed`) for
the fields that are actually reactive here, at the cost of every
reactive field needing its own explicit property pair instead of every
bare attribute write being automatically interceptable.
"""
from __future__ import annotations

from pugtk._vector import Vector3
from pugtk._matrix import Matrix4
from pugtk._mesh import Mesh
from pugtk._texture import Texture

from ._signal import Signal


GRAVITY: float = -9.8


class Instance:
    name: str
    mesh: Mesh
    colors: list[int]
    texture: Texture
    parent: "Instance"
    children: list

    changed: Signal
    touched: Signal

    velocity: Vector3
    gravity_enabled: int
    anchored: int

    def __init__(self, name: str, mesh: Mesh, colors: list[int]) -> None:
        self.name = name
        self.mesh = mesh
        self.colors = colors
        self.texture = None
        self.parent = None
        self.children = []

        self._position: Vector3 = Vector3(0.0, 0.0, 0.0)
        self._rotation: Vector3 = Vector3(0.0, 0.0, 0.0)
        self._scale: Vector3 = Vector3(1.0, 1.0, 1.0)
        self._model_dirty: int = 1
        self._model_cache: Matrix4 = Matrix4.identity()

        self.velocity = Vector3(0.0, 0.0, 0.0)
        self.gravity_enabled = 0
        self.anchored = 0

        self.changed = Signal()
        self.touched = Signal()

    @property
    def position(self) -> Vector3:
        return self._position

    @position.setter
    def position(self, value: Vector3) -> None:
        self._position = value
        self._model_dirty = 1
        self.changed(0)

    @property
    def rotation(self) -> Vector3:
        """Euler angles in radians (x, y, z)."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Vector3) -> None:
        self._rotation = value
        self._model_dirty = 1
        self.changed(1)

    @property
    def scale(self) -> Vector3:
        return self._scale

    @scale.setter
    def scale(self, value: Vector3) -> None:
        self._scale = value
        self._model_dirty = 1
        self.changed(2)

    def physics_step(self, dt: float) -> None:
        """Integrates velocity into position. If gravity_enabled=1 and the
        Instance is not anchored, GRAVITY (m/s²) accelerates the y axis.
        Anchored Instances are never moved by physics (they act as static
        colliders -- walls, floors, obstacles)."""
        if self.anchored == 1:
            return
        if self.gravity_enabled == 1:
            self.velocity = Vector3(
                self.velocity.x,
                self.velocity.y + GRAVITY * dt,
                self.velocity.z,
            )
        new_pos: Vector3 = Vector3(
            self._position.x + self.velocity.x * dt,
            self._position.y + self.velocity.y * dt,
            self._position.z + self.velocity.z * dt,
        )
        self.position = new_pos

    def add_child(self, child: "Instance") -> None:
        child.parent = self
        self.children.append(child)

    def local_matrix(self) -> Matrix4:
        """Position * RotationZ * RotationY * RotationX * Scale, recomputed
        only when a property write marked the cache dirty (most frames
        touch only a handful of instances, so this avoids redoing the
        4 matrix multiplies for every static prop every frame)."""
        if self._model_dirty == 0:
            return self._model_cache
        t: Matrix4 = Matrix4.translation(self._position.x, self._position.y, self._position.z)
        rz: Matrix4 = Matrix4.rotation_z(self._rotation.z)
        ry: Matrix4 = Matrix4.rotation_y(self._rotation.y)
        rx: Matrix4 = Matrix4.rotation_x(self._rotation.x)
        s: Matrix4 = Matrix4.scale(self._scale.x, self._scale.y, self._scale.z)
        m: Matrix4 = t.multiply(rz).multiply(ry).multiply(rx).multiply(s)
        self._model_cache = m
        self._model_dirty = 0
        return m

    def world_matrix(self) -> Matrix4:
        """local_matrix(), composed with every ancestor's local_matrix()
        up the parent chain -- a child's position/rotation/scale are
        relative to its parent, matching Roblox's Model/Part nesting."""
        local: Matrix4 = self.local_matrix()
        if self.parent is None:
            return local
        parent_world: Matrix4 = self.parent.world_matrix()
        return parent_world.multiply(local)
