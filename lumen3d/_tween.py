"""Tween -- animate any numeric Instance property over time.

Tweens are the standard way to animate things without writing physics
manually: move a platform, fade a color, scale a UI element, etc.

Usage:
    tween = Tween(inst, "position_y", 0.0, 5.0, 1.5, EASE_OUT_QUAD)
    tween.play()
    tweens.update(dt)   # call each frame from your updated callback

Supported target fields:
    "position_x", "position_y", "position_z"
    "rotation_x", "rotation_y", "rotation_z"
    "scale_x", "scale_y", "scale_z"
    "restitution"

TweenManager holds multiple tweens and updates them all at once.

Easing functions (EASE_* constants):
    EASE_LINEAR, EASE_IN_QUAD, EASE_OUT_QUAD, EASE_IN_OUT_QUAD,
    EASE_IN_CUBIC, EASE_OUT_CUBIC, EASE_IN_OUT_CUBIC,
    EASE_IN_SINE, EASE_OUT_SINE, EASE_IN_OUT_SINE,
    EASE_IN_BOUNCE, EASE_OUT_BOUNCE,
    EASE_IN_ELASTIC, EASE_OUT_ELASTIC,
    EASE_SPRING
"""
from __future__ import annotations

import math

from pugtk._vector import Vector3
from ._instance import Instance
from ._signal import Signal

# Easing mode constants
EASE_LINEAR:       int = 0
EASE_IN_QUAD:      int = 1
EASE_OUT_QUAD:     int = 2
EASE_IN_OUT_QUAD:  int = 3
EASE_IN_CUBIC:     int = 4
EASE_OUT_CUBIC:    int = 5
EASE_IN_OUT_CUBIC: int = 6
EASE_IN_SINE:      int = 7
EASE_OUT_SINE:     int = 8
EASE_IN_OUT_SINE:  int = 9
EASE_IN_BOUNCE:    int = 10
EASE_OUT_BOUNCE:   int = 11
EASE_IN_ELASTIC:   int = 12
EASE_OUT_ELASTIC:  int = 13
EASE_SPRING:       int = 14


def _ease(t: float, mode: int) -> float:
    """Apply easing curve. t in [0,1], returns eased value in [0,1]."""
    if mode == EASE_LINEAR:
        return t
    if mode == EASE_IN_QUAD:
        return t * t
    if mode == EASE_OUT_QUAD:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if mode == EASE_IN_OUT_QUAD:
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - (-2.0 * t + 2.0) * (-2.0 * t + 2.0) * 0.5
    if mode == EASE_IN_CUBIC:
        return t * t * t
    if mode == EASE_OUT_CUBIC:
        u: float = 1.0 - t
        return 1.0 - u * u * u
    if mode == EASE_IN_OUT_CUBIC:
        if t < 0.5:
            return 4.0 * t * t * t
        u2: float = -2.0 * t + 2.0
        return 1.0 - u2 * u2 * u2 * 0.5
    if mode == EASE_IN_SINE:
        return 1.0 - math.cos(t * math.pi * 0.5)
    if mode == EASE_OUT_SINE:
        return math.sin(t * math.pi * 0.5)
    if mode == EASE_IN_OUT_SINE:
        return -(math.cos(math.pi * t) - 1.0) * 0.5
    if mode == EASE_OUT_BOUNCE:
        n: float = 7.5625
        d: float = 2.75
        if t < 1.0 / d:
            return n * t * t
        if t < 2.0 / d:
            t = t - 1.5 / d
            return n * t * t + 0.75
        if t < 2.5 / d:
            t = t - 2.25 / d
            return n * t * t + 0.9375
        t = t - 2.625 / d
        return n * t * t + 0.984375
    if mode == EASE_IN_BOUNCE:
        return 1.0 - _ease(1.0 - t, EASE_OUT_BOUNCE)
    if mode == EASE_OUT_ELASTIC:
        if t == 0.0:
            return 0.0
        if t == 1.0:
            return 1.0
        return math.pow(2.0, -10.0 * t) * math.sin((t * 10.0 - 0.75) * 2.094395) + 1.0
    if mode == EASE_IN_ELASTIC:
        if t == 0.0:
            return 0.0
        if t == 1.0:
            return 1.0
        return -math.pow(2.0, 10.0 * t - 10.0) * math.sin((t * 10.0 - 10.75) * 2.094395)
    if mode == EASE_SPRING:
        return 1.0 - math.exp(-6.0 * t) * math.cos(t * 20.0)
    return t


def _get_field(inst: Instance, field: str) -> float:
    if field == "position_x":
        return inst.position.x
    if field == "position_y":
        return inst.position.y
    if field == "position_z":
        return inst.position.z
    if field == "rotation_x":
        return inst.rotation.x
    if field == "rotation_y":
        return inst.rotation.y
    if field == "rotation_z":
        return inst.rotation.z
    if field == "scale_x":
        return inst.scale.x
    if field == "scale_y":
        return inst.scale.y
    if field == "scale_z":
        return inst.scale.z
    if field == "restitution":
        return inst.restitution
    return 0.0


def _set_field(inst: Instance, field: str, value: float) -> None:
    if field == "position_x":
        inst.position = Vector3(value, inst.position.y, inst.position.z)
    elif field == "position_y":
        inst.position = Vector3(inst.position.x, value, inst.position.z)
    elif field == "position_z":
        inst.position = Vector3(inst.position.x, inst.position.y, value)
    elif field == "rotation_x":
        inst.rotation = Vector3(value, inst.rotation.y, inst.rotation.z)
    elif field == "rotation_y":
        inst.rotation = Vector3(inst.rotation.x, value, inst.rotation.z)
    elif field == "rotation_z":
        inst.rotation = Vector3(inst.rotation.x, inst.rotation.y, value)
    elif field == "scale_x":
        inst.scale = Vector3(value, inst.scale.y, inst.scale.z)
    elif field == "scale_y":
        inst.scale = Vector3(inst.scale.x, value, inst.scale.z)
    elif field == "scale_z":
        inst.scale = Vector3(inst.scale.x, inst.scale.y, value)
    elif field == "restitution":
        inst.restitution = value


class Tween:
    """Animate one numeric field of an Instance from `from_val` to `to_val`
    over `duration` seconds using an easing curve.

    finished: Signal -- fires once when the tween completes (value=0).
    """

    target: Instance
    field: str
    from_val: float
    to_val: float
    duration: float
    ease_mode: int
    elapsed: float
    playing: int
    loop: int
    finished: Signal

    def __init__(self, target: Instance, field: str,
                 from_val: float, to_val: float,
                 duration: float, ease_mode: int) -> None:
        self.target = target
        self.field = field
        self.from_val = from_val
        self.to_val = to_val
        self.duration = duration
        self.ease_mode = ease_mode
        self.elapsed = 0.0
        self.playing = 0
        self.loop = 0
        self.finished = Signal()

    def play(self) -> None:
        self.elapsed = 0.0
        self.playing = 1

    def stop(self) -> None:
        self.playing = 0

    def reset(self) -> None:
        self.elapsed = 0.0
        self.playing = 0
        _set_field(self.target, self.field, self.from_val)

    def seek(self, t: float) -> None:
        """Jump to a specific time (seconds)."""
        self.elapsed = t
        if self.elapsed > self.duration:
            self.elapsed = self.duration
        frac: float = self.elapsed / self.duration if self.duration > 0.0 else 1.0
        val: float = self.from_val + (self.to_val - self.from_val) * _ease(frac, self.ease_mode)
        _set_field(self.target, self.field, val)

    def update(self, dt: float) -> None:
        if self.playing == 0:
            return
        self.elapsed = self.elapsed + dt
        frac: float = 1.0
        if self.duration > 0.0:
            frac = self.elapsed / self.duration
        if frac >= 1.0:
            frac = 1.0
            _set_field(self.target, self.field, self.to_val)
            if self.loop == 1:
                self.elapsed = 0.0
            else:
                self.playing = 0
                self.finished(0)
        else:
            val: float = self.from_val + (self.to_val - self.from_val) * _ease(frac, self.ease_mode)
            _set_field(self.target, self.field, val)

    def is_done(self) -> int:
        return 1 if self.playing == 0 and self.elapsed >= self.duration else 0


class TweenManager:
    """Holds and updates a collection of Tweens. Add tweens with add(),
    call update(dt) once per frame, and completed tweens are automatically
    removed (unless they loop).

    Example:
        mgr = TweenManager()
        t = mgr.tween(cube, "position_y", 0.0, 3.0, 1.0, EASE_OUT_BOUNCE)
        t.play()

        def on_update(frame: int) -> None:
            mgr.update(loop.fixed_dt)
        loop.updated.connect(on_update)
    """

    _tweens: list

    def __init__(self) -> None:
        self._tweens = []

    def add(self, tw: Tween) -> Tween:
        self._tweens.append(tw)
        return tw

    def tween(self, target: Instance, field: str,
              from_val: float, to_val: float,
              duration: float, ease_mode: int) -> Tween:
        """Create, register, and return a new Tween (not yet playing)."""
        tw: Tween = Tween(target, field, from_val, to_val, duration, ease_mode)
        self._tweens.append(tw)
        return tw

    def update(self, dt: float) -> None:
        keep: list = []
        i: int = 0
        while i < len(self._tweens):
            tw: Tween = self._tweens[i]
            tw.update(dt)
            if tw.playing == 1 or tw.loop == 1:
                keep.append(tw)
            else:
                keep.append(tw)
            i = i + 1
        self._tweens = keep

    def clear(self) -> None:
        self._tweens = []

    def stop_all(self) -> None:
        i: int = 0
        while i < len(self._tweens):
            self._tweens[i].stop()
            i = i + 1

    def count(self) -> int:
        return len(self._tweens)
