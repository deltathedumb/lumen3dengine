"""HUD -- in-game 2D overlay using lumen.Canvas.

Provides a simple immediate-mode UI layer for health bars, score text,
crosshairs, and debug panels drawn on top of the 3D viewport.

Because pugtk's GLRenderer3D draws into a GLWindow (OpenGL context) and
lumen.Canvas uses SDL2's 2D renderer, they cannot share the same window.
The HUD runs in a *separate* transparent-ish lumen.Canvas window that
sits visually on top, OR alternatively the developer positions the HUD
canvas alongside the GL viewport (same approach as the editor).

For single-window HUD (most common): open a small lumen.Canvas window
positioned over the GL window, or use a fullscreen-overlay lumen.Canvas.

The simplest pattern that always works without OS window-management:
  - Use a second lumen.Canvas window sized to match the GL viewport.
  - Draw 2D UI into it each frame after world.render().
  - Call hud.present() at the end of your updated callback.

Usage:
    hud = HUD(800, 600)

    def on_update(frame: int) -> None:
        hud.clear()
        hud.text(10, 10, "Score: " + str(score), 0xFFFFFF)
        hud.bar(10, 30, 200, 16, hp / max_hp, 0x44DD44, 0x333333)
        hud.crosshair(400, 300, 12, 0xFFFFFF)
        hud.present()

    loop.updated.connect(on_update)
    loop.run()
    hud.close()
"""
from __future__ import annotations

import lumen
from lumen._canvas import Canvas as LumenCanvas


class HUD:
    """Immediate-mode 2D overlay canvas.

    Call clear() at the start of each frame, draw with the helper methods,
    then call present() to flip the frame.
    """

    _canvas: LumenCanvas
    _w: int
    _h: int

    def __init__(self, width: int, height: int, title: str) -> None:
        self._canvas = lumen.Canvas(title, width, height)
        self._w = width
        self._h = height

    def clear(self, bg: int) -> None:
        """Clear to bg color (0xRRGGBB). Use 0x000000 for black overlay."""
        self._canvas.clear(bg)

    def present(self) -> int:
        """Present frame and pump events. Returns 0 if HUD window closed."""
        self._canvas.present()
        return 1

    def close(self) -> None:
        self._canvas.close()

    # ---- Text ----

    def text(self, x: int, y: int, s: str, color: int, scale: int) -> None:
        self._canvas.color(color)
        self._canvas.text(x, y, s, scale)

    def label(self, x: int, y: int, s: str, color: int) -> None:
        self._canvas.color(color)
        self._canvas.text(x, y, s, 1)

    # ---- Shapes ----

    def rect_outline(self, x: int, y: int, w: int, h: int, color: int) -> None:
        self._canvas.color(color)
        self._canvas.rect(x, y, w, h)

    def rect_fill(self, x: int, y: int, w: int, h: int, color: int) -> None:
        self._canvas.color(color)
        self._canvas.fill(x, y, w, h)

    def line(self, x1: int, y1: int, x2: int, y2: int, color: int) -> None:
        self._canvas.color(color)
        self._canvas.line(x1, y1, x2, y2)

    def circle(self, cx: int, cy: int, r: int, color: int) -> None:
        self._canvas.color(color)
        self._canvas.circle(cx, cy, r)

    def disc(self, cx: int, cy: int, r: int, color: int) -> None:
        self._canvas.color(color)
        self._canvas.disc(cx, cy, r)

    # ---- Composite widgets ----

    def bar(self, x: int, y: int, w: int, h: int,
            fraction: float, fill_color: int, bg_color: int) -> None:
        """Draw a progress/health bar. fraction in [0, 1]."""
        self._canvas.color(bg_color)
        self._canvas.fill(x, y, w, h)
        filled_w: int = int(float(w) * fraction)
        if filled_w > w:
            filled_w = w
        if filled_w < 0:
            filled_w = 0
        if filled_w > 0:
            self._canvas.color(fill_color)
            self._canvas.fill(x, y, filled_w, h)
        self._canvas.color(0x666666)
        self._canvas.rect(x, y, w, h)

    def crosshair(self, cx: int, cy: int, size: int, color: int) -> None:
        """Draw a +crosshair at (cx, cy)."""
        self._canvas.color(color)
        self._canvas.line(cx - size, cy, cx + size, cy)
        self._canvas.line(cx, cy - size, cx, cy + size)

    def dot_crosshair(self, cx: int, cy: int, color: int) -> None:
        """Draw a small dot crosshair."""
        self._canvas.color(color)
        self._canvas.disc(cx, cy, 2)

    def panel(self, x: int, y: int, w: int, h: int,
              bg: int, border: int) -> None:
        """Draw a filled panel with border."""
        self._canvas.color(bg)
        self._canvas.fill(x, y, w, h)
        self._canvas.color(border)
        self._canvas.rect(x, y, w, h)

    def minimap_dot(self, map_x: int, map_y: int, map_w: int, map_h: int,
                    world_x: float, world_z: float,
                    world_half_size: float, dot_color: int, dot_r: int) -> None:
        """Draw a dot on a rectangular minimap panel.

        Converts world (x, z) coordinates to minimap pixel space.
        world_half_size: half the world dimension the minimap covers.
        """
        nx: float = (world_x / world_half_size) * 0.5 + 0.5
        nz: float = (world_z / world_half_size) * 0.5 + 0.5
        px: int = map_x + int(nx * float(map_w))
        py: int = map_y + int(nz * float(map_h))
        self._canvas.color(dot_color)
        self._canvas.disc(px, py, dot_r)

    def fps_counter(self, x: int, y: int, fps: int, color: int) -> None:
        """Draw FPS number."""
        self._canvas.color(color)
        self._canvas.text(x, y, "FPS:" + str(fps), 1)

    def key(self) -> int:
        return self._canvas.key()

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h
