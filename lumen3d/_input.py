"""Input -- direct keyboard/mouse state polling, the lumen3dengine
equivalent of Roblox's UserInputService for the common "is this key held
right now" case scripts actually want most often (movement, held-button
actions) rather than queued discrete key-press events.

Built on _gui_sdl's existing is_key_down()/mouse_x()/mouse_y()/
mouse_button() FFI bindings (already-working SDL_GetKeyboardState/
SDL_GetMouseState wrappers) -- this class only adds the ergonomic layer
GameLoop wires in automatically, so a script never touches _gui_sdl
directly. Key codes are SDL scancodes (_gui_sdl.KEY_W, KEY_SPACE, ...);
re-exported here as KEY_* constants so a script importing lumen3d alone
(not pugtk/_gui_sdl) has everything it needs.
"""
from __future__ import annotations

import _gui_sdl


KEY_A = _gui_sdl.KEY_A
KEY_B = _gui_sdl.KEY_B
KEY_C = _gui_sdl.KEY_C
KEY_D = _gui_sdl.KEY_D
KEY_E = _gui_sdl.KEY_E
KEY_F = _gui_sdl.KEY_F
KEY_G = _gui_sdl.KEY_G
KEY_H = _gui_sdl.KEY_H
KEY_I = _gui_sdl.KEY_I
KEY_J = _gui_sdl.KEY_J
KEY_K = _gui_sdl.KEY_K
KEY_L = _gui_sdl.KEY_L
KEY_M = _gui_sdl.KEY_M
KEY_N = _gui_sdl.KEY_N
KEY_O = _gui_sdl.KEY_O
KEY_P = _gui_sdl.KEY_P
KEY_Q = _gui_sdl.KEY_Q
KEY_R = _gui_sdl.KEY_R
KEY_S = _gui_sdl.KEY_S
KEY_T = _gui_sdl.KEY_T
KEY_U = _gui_sdl.KEY_U
KEY_V = _gui_sdl.KEY_V
KEY_W = _gui_sdl.KEY_W
KEY_X = _gui_sdl.KEY_X
KEY_Y = _gui_sdl.KEY_Y
KEY_Z = _gui_sdl.KEY_Z
KEY_SPACE = _gui_sdl.KEY_SPACE
KEY_RETURN = _gui_sdl.KEY_RETURN
KEY_ESCAPE = _gui_sdl.KEY_ESCAPE
KEY_LEFT = _gui_sdl.KEY_LEFT
KEY_RIGHT = _gui_sdl.KEY_RIGHT
KEY_UP = _gui_sdl.KEY_UP
KEY_DOWN = _gui_sdl.KEY_DOWN
KEY_LSHIFT = _gui_sdl.KEY_LSHIFT
KEY_LCTRL = _gui_sdl.KEY_LCTRL

MOUSE_LEFT = 1
MOUSE_MIDDLE = 2
MOUSE_RIGHT = 3


class Input:
    mouse_x: int
    mouse_y: int

    def __init__(self) -> None:
        self.mouse_x = 0
        self.mouse_y = 0

    def is_key_down(self, keycode: int) -> int:
        return _gui_sdl.is_key_down(keycode)

    def is_mouse_down(self, button: int) -> int:
        current: int = _gui_sdl.mouse_button()
        return 1 if current == button else 0

    def refresh(self) -> None:
        """Called once per frame by GameLoop before firing `updated` --
        snapshots mouse position so a script reading input.mouse_x/
        mouse_y mid-frame sees a stable value instead of re-querying SDL
        (whose state can change between two reads within the same
        frame if the OS delivers a motion event in between)."""
        self.mouse_x = _gui_sdl.mouse_x()
        self.mouse_y = _gui_sdl.mouse_y()
