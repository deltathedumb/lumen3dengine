"""Particle system for lumen3dengine.

Emits, updates, and renders CPU-side particle effects. Each particle is
a small Instance (sphere or cube) spawned, aged, and recycled by the
emitter.  Because asmpython AOT-compiled Instances are already fast,
this approach works for moderate particle counts (up to ~200 particles
per emitter before frame budgets start pinching).

For larger counts, particles are rendered as tiny colored cubes sharing
one mesh -- each particle gets its own Instance so the existing World
rendering pipeline handles them transparently.

Usage:
    emitter = ParticleEmitter(world, "Sparks")
    emitter.emit_rate = 10.0        # particles per second
    emitter.lifetime = 0.8          # seconds per particle
    emitter.speed = 4.0             # initial speed
    emitter.spread = 1.2            # cone half-angle radians
    emitter.color_start = 0xFFCC44
    emitter.color_end = 0xFF4400
    emitter.scale_start = 0.15
    emitter.scale_end = 0.02
    emitter.gravity = 1             # use world gravity

    emitter.position = Vector3(0.0, 2.0, 0.0)
    emitter.direction = Vector3(0.0, 1.0, 0.0)
    emitter.playing = 1

    def on_update(frame: int) -> None:
        emitter.update(loop.fixed_dt)

    loop.updated.connect(on_update)
"""
from __future__ import annotations

import math

from pugtk._vector import Vector3
from pugtk._mesh import Mesh

from ._instance import Instance, GRAVITY
from ._world import World
from ._material import Material


def _lerp_color(a: int, b: int, t: float) -> int:
    ar: int = (a >> 16) & 0xFF
    ag: int = (a >> 8) & 0xFF
    ab: int = a & 0xFF
    br: int = (b >> 16) & 0xFF
    bg_: int = (b >> 8) & 0xFF
    bb: int = b & 0xFF
    rr: int = int(float(ar) + (float(br) - float(ar)) * t)
    rg: int = int(float(ag) + (float(bg_) - float(ag)) * t)
    rb: int = int(float(ab) + (float(bb) - float(ab)) * t)
    return (rr << 16) | (rg << 8) | rb


def _rand_float(seed: int) -> float:
    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return float(seed & 0xFFFF) / 65535.0


class _Particle:
    inst: Instance
    vel_x: float
    vel_y: float
    vel_z: float
    age: float
    lifetime: float
    alive: int
    seed: int

    def __init__(self, inst: Instance) -> None:
        self.inst = inst
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.age = 0.0
        self.lifetime = 1.0
        self.alive = 0
        self.seed = 0


class ParticleEmitter:
    """Spawns and manages a pool of particle Instances.

    All particle Instances are pre-allocated in __init__ and recycled --
    no per-frame allocation.
    """

    _world: World
    _pool: list
    _mesh: Mesh
    _emit_accum: float

    position: Vector3
    direction: Vector3

    emit_rate: float
    lifetime: float
    speed: float
    speed_variance: float
    spread: float
    gravity: int

    color_start: int
    color_end: int
    scale_start: float
    scale_end: float

    playing: int
    one_shot: int
    _seed: int

    def __init__(self, world: World, name: str, pool_size: int) -> None:
        self._world = world
        self._pool = []
        self._mesh = Mesh.cube(1.0)
        self._emit_accum = 0.0
        self._seed = 12345

        self.position = Vector3(0.0, 0.0, 0.0)
        self.direction = Vector3(0.0, 1.0, 0.0)

        self.emit_rate = 10.0
        self.lifetime = 1.0
        self.speed = 3.0
        self.speed_variance = 0.5
        self.spread = 0.5
        self.gravity = 0

        self.color_start = 0xFFFFFF
        self.color_end = 0xFF4400
        self.scale_start = 0.15
        self.scale_end = 0.02

        self.playing = 0
        self.one_shot = 0

        mat: Material = Material(name)
        mat.color = self.color_start
        i: int = 0
        while i < pool_size:
            pname: str = name + "_p" + str(i)
            inst: Instance = Instance(pname, self._mesh, [])
            inst.set_material(mat)
            inst.anchored = 1
            inst.scale = Vector3(0.0, 0.0, 0.0)
            world.add(inst)
            p: _Particle = _Particle(inst)
            self._pool.append(p)
            i = i + 1

    def _next_rand(self) -> float:
        self._seed = (self._seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return float(self._seed & 0xFFFF) / 65535.0

    def _spawn_one(self) -> None:
        p: _Particle = None
        i: int = 0
        while i < len(self._pool):
            candidate: _Particle = self._pool[i]
            if candidate.alive == 0:
                p = candidate
                break
            i = i + 1
        if p is None:
            return

        p.alive = 1
        p.age = 0.0
        p.lifetime = self.lifetime

        p.inst.position = self.position

        spd: float = self.speed + (self._next_rand() - 0.5) * self.speed_variance * 2.0
        theta: float = (self._next_rand() - 0.5) * self.spread * 2.0
        phi: float = self._next_rand() * 6.2832

        perp_x: float = self.direction.z
        perp_z: float = -self.direction.x
        perp_len: float = math.sqrt(perp_x * perp_x + perp_z * perp_z)
        if perp_len < 0.0001:
            perp_x = 1.0
            perp_z = 0.0
            perp_len = 1.0
        perp_x = perp_x / perp_len
        perp_z = perp_z / perp_len

        t_off: float = math.tan(theta)
        radial_x: float = perp_x * math.cos(phi) * t_off
        radial_z: float = perp_z * math.cos(phi) * t_off

        p.vel_x = (self.direction.x + radial_x) * spd
        p.vel_y = self.direction.y * spd
        p.vel_z = (self.direction.z + radial_z) * spd

        p.inst.scale = Vector3(self.scale_start, self.scale_start, self.scale_start)

    def update(self, dt: float) -> None:
        if self.playing == 1:
            self._emit_accum = self._emit_accum + dt * self.emit_rate
            while self._emit_accum >= 1.0:
                self._spawn_one()
                self._emit_accum = self._emit_accum - 1.0

        i: int = 0
        while i < len(self._pool):
            p: _Particle = self._pool[i]
            if p.alive == 1:
                p.age = p.age + dt
                if p.age >= p.lifetime:
                    p.alive = 0
                    p.inst.scale = Vector3(0.0, 0.0, 0.0)
                else:
                    t: float = p.age / p.lifetime
                    if self.gravity == 1:
                        p.vel_y = p.vel_y + GRAVITY * dt
                    new_x: float = p.inst.position.x + p.vel_x * dt
                    new_y: float = p.inst.position.y + p.vel_y * dt
                    new_z: float = p.inst.position.z + p.vel_z * dt
                    p.inst.position = Vector3(new_x, new_y, new_z)
                    sc: float = self.scale_start + (self.scale_end - self.scale_start) * t
                    p.inst.scale = Vector3(sc, sc, sc)
                    col: int = _lerp_color(self.color_start, self.color_end, t)
                    mat: Material = Material("p")
                    mat.color = col
                    p.inst.set_material(mat)
            i = i + 1

    def burst(self, count: int) -> None:
        """Emit `count` particles immediately (one-shot burst)."""
        i: int = 0
        while i < count:
            self._spawn_one()
            i = i + 1

    def stop(self) -> None:
        self.playing = 0

    def play(self) -> None:
        self.playing = 1

    def kill_all(self) -> None:
        """Immediately kill all live particles."""
        i: int = 0
        while i < len(self._pool):
            self._pool[i].alive = 0
            self._pool[i].inst.scale = Vector3(0.0, 0.0, 0.0)
            i = i + 1

    def alive_count(self) -> int:
        n: int = 0
        i: int = 0
        while i < len(self._pool):
            if self._pool[i].alive == 1:
                n = n + 1
            i = i + 1
        return n
