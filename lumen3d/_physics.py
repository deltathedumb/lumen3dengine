"""Extended physics properties for Instances.

Adds per-Instance friction, drag (linear damping), and impulse API.
Also provides PhysicsWorld which replaces World._resolve_collision with
a more complete solver supporting friction and drag.

Usage:
    # Apply a one-shot impulse (e.g. explosion knockback):
    apply_impulse(inst, Vector3(0.0, 8.0, 3.0))

    # Set per-object drag so objects slow naturally:
    inst.drag = 0.98       # multiply velocity by this each frame (0..1)
    inst.friction = 0.85   # horizontal damping when grounded

    # These are plain float attributes -- add them to an Instance manually:
    inst.drag = 0.99
    inst.friction = 0.8

These can be set directly on any Instance since Instance allows arbitrary
attribute assignment. The physics helpers below read them if present,
or use default values if absent.
"""
from __future__ import annotations

import math

from pugtk._vector import Vector3
from ._instance import Instance


DEFAULT_DRAG:     float = 1.0
DEFAULT_FRICTION: float = 1.0


def get_drag(inst: Instance) -> float:
    """Return inst.drag if it has been set, else DEFAULT_DRAG."""
    v = getattr(inst, "drag", None)
    if v is None:
        return DEFAULT_DRAG
    return float(v)


def get_friction(inst: Instance) -> float:
    v = getattr(inst, "friction", None)
    if v is None:
        return DEFAULT_FRICTION
    return float(v)


def apply_impulse(inst: Instance, impulse: Vector3) -> None:
    """Add a one-frame velocity kick to inst. Ignores anchored flag
    so you can nudge anchored objects programmatically if desired."""
    inst.velocity = Vector3(
        inst.velocity.x + impulse.x,
        inst.velocity.y + impulse.y,
        inst.velocity.z + impulse.z,
    )


def apply_force(inst: Instance, force: Vector3, dt: float) -> None:
    """Integrate a force (mass assumed = 1) over dt into velocity."""
    inst.velocity = Vector3(
        inst.velocity.x + force.x * dt,
        inst.velocity.y + force.y * dt,
        inst.velocity.z + force.z * dt,
    )


def apply_drag(inst: Instance, dt: float) -> None:
    """Apply linear damping -- multiply velocity by drag^dt so the
    time constant is independent of frame rate."""
    drag: float = get_drag(inst)
    if drag >= 1.0:
        return
    factor: float = math.pow(drag, dt * 60.0)
    inst.velocity = Vector3(
        inst.velocity.x * factor,
        inst.velocity.y * factor,
        inst.velocity.z * factor,
    )


def apply_friction(inst: Instance) -> None:
    """Apply horizontal friction after a collision grounds the object.
    Call once per frame after collision resolution when the object is
    resting on a surface (velocity.y ~= 0)."""
    friction: float = get_friction(inst)
    if friction >= 1.0:
        return
    inst.velocity = Vector3(
        inst.velocity.x * friction,
        inst.velocity.y,
        inst.velocity.z * friction,
    )


def speed(inst: Instance) -> float:
    """Return current speed (magnitude of velocity)."""
    vx: float = inst.velocity.x
    vy: float = inst.velocity.y
    vz: float = inst.velocity.z
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def clamp_speed(inst: Instance, max_speed: float) -> None:
    """Cap velocity magnitude to max_speed."""
    s: float = speed(inst)
    if s > max_speed and s > 0.0:
        scale: float = max_speed / s
        inst.velocity = Vector3(
            inst.velocity.x * scale,
            inst.velocity.y * scale,
            inst.velocity.z * scale,
        )


def is_grounded(inst: Instance, threshold: float) -> int:
    """Return 1 if |velocity.y| < threshold (heuristic for being on ground)."""
    vy: float = inst.velocity.y
    if vy < 0.0:
        vy = -vy
    return 1 if vy < threshold else 0


def distance(a: Instance, b: Instance) -> float:
    """Return the Euclidean distance between two instances' positions."""
    dx: float = b.position.x - a.position.x
    dy: float = b.position.y - a.position.y
    dz: float = b.position.z - a.position.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def direction_to(from_inst: Instance, to_inst: Instance) -> Vector3:
    """Return the normalized direction vector from from_inst to to_inst."""
    dx: float = to_inst.position.x - from_inst.position.x
    dy: float = to_inst.position.y - from_inst.position.y
    dz: float = to_inst.position.z - from_inst.position.z
    mag: float = math.sqrt(dx * dx + dy * dy + dz * dz)
    if mag < 0.0001:
        return Vector3(0.0, 0.0, 0.0)
    return Vector3(dx / mag, dy / mag, dz / mag)


def look_at_y(inst: Instance, target: Instance) -> None:
    """Rotate inst.rotation.y so it faces target (horizontal only)."""
    dx: float = target.position.x - inst.position.x
    dz: float = target.position.z - inst.position.z
    angle: float = math.atan2(dx, dz)
    inst.rotation = Vector3(inst.rotation.x, angle, inst.rotation.z)
