"""Debug overlay utilities for lumen3dengine.

Provides:
  DebugStats  -- track FPS, frame time
  DebugLog    -- in-memory log ring buffer, draw to HUD
  DebugTimer  -- simple stopwatch for CPU profiling

Usage:
    stats = DebugStats(30)
    log = DebugLog(20)

    def on_update(frame: int) -> None:
        stats.tick(loop.fixed_dt)
        log.info("vel: " + str(player.velocity.y))
        hud.clear(0)
        stats.draw(hud, 4, 4)
        log.draw(hud, 4, 60)
        hud.present()

    loop.updated.connect(on_update)
"""
from __future__ import annotations

from ._hud import HUD


class DebugStats:
    """Track frame rate and frame time.

    tick(dt) must be called once per frame with the fixed_dt value.
    """

    _frame_times: list
    _ft_head: int
    _ft_count: int
    frame: int
    fps: int

    def __init__(self, history: int) -> None:
        self._frame_times = []
        self._ft_head = 0
        self._ft_count = history
        self.frame = 0
        self.fps = 0
        i: int = 0
        while i < history:
            self._frame_times.append(0.016)
            i = i + 1

    def tick(self, dt: float) -> None:
        self._frame_times[self._ft_head] = dt
        self._ft_head = (self._ft_head + 1) % self._ft_count
        self.frame = self.frame + 1
        avg: float = 0.0
        i: int = 0
        while i < self._ft_count:
            avg = avg + self._frame_times[i]
            i = i + 1
        avg = avg / float(self._ft_count)
        if avg > 0.0:
            self.fps = int(1.0 / avg)
        else:
            self.fps = 0

    def draw(self, hud: HUD, x: int, y: int) -> None:
        hud.text(x, y, "FPS: " + str(self.fps), 0x44FF44, 1)
        hud.text(x, y + 10, "frame: " + str(self.frame), 0x888888, 1)


class DebugLog:
    """Fixed-capacity ring buffer of log lines, drawn to a HUD."""

    _lines: list
    _head: int
    _capacity: int

    def __init__(self, capacity: int) -> None:
        self._lines = []
        self._head = 0
        self._capacity = capacity
        i: int = 0
        while i < capacity:
            self._lines.append("")
            i = i + 1

    def info(self, msg: str) -> None:
        self._lines[self._head] = msg
        self._head = (self._head + 1) % self._capacity

    def warn(self, msg: str) -> None:
        self._lines[self._head] = "[WARN] " + msg
        self._head = (self._head + 1) % self._capacity

    def error(self, msg: str) -> None:
        self._lines[self._head] = "[ERR]  " + msg
        self._head = (self._head + 1) % self._capacity

    def clear(self) -> None:
        i: int = 0
        while i < self._capacity:
            self._lines[i] = ""
            i = i + 1
        self._head = 0

    def draw(self, hud: HUD, x: int, y: int) -> None:
        i: int = 0
        while i < self._capacity:
            idx: int = (self._head + i) % self._capacity
            line: str = self._lines[idx]
            if line != "":
                col: int = 0xCCCCCC
                if len(line) >= 6 and line[0:6] == "[WARN]":
                    col = 0xFFCC44
                elif len(line) >= 5 and line[0:5] == "[ERR]":
                    col = 0xFF4444
                hud.text(x, y + i * 10, line, col, 1)
            i = i + 1


class DebugTimer:
    """Simple stopwatch for profiling blocks of code (CPU side only)."""

    _start: float
    _elapsed: float
    label: str

    def __init__(self, label: str) -> None:
        self.label = label
        self._start = 0.0
        self._elapsed = 0.0

    def start(self, ticks: int) -> None:
        self._start = float(ticks)

    def stop(self, ticks: int) -> None:
        self._elapsed = float(ticks) - self._start

    def ms(self) -> float:
        return self._elapsed

    def draw(self, hud: HUD, x: int, y: int) -> None:
        ms_str: str = str(int(self._elapsed))
        hud.text(x, y, self.label + ": " + ms_str + "ms", 0xAADDFF, 1)
