"""GameLoop -- replaces every example's hand-rolled `while running:
render_scene(...); running = renderer.update()` boilerplate with a
`run()` call plus registered per-frame update callbacks, the way a real
engine's scripting layer doesn't make every script author re-derive their
own main loop.
"""
from __future__ import annotations

from pugtk._renderer3d_gl import GLWindow

from ._world import World
from ._signal import Signal
from ._input import Input
from ._instance import Instance


class GameLoop:
    window: GLWindow
    world: World
    input: Input
    updated: Signal
    frame_count: int
    fixed_dt: float

    def __init__(self, window: GLWindow, world: World) -> None:
        self.window = window
        self.world = world
        self.input = Input()
        self.updated = Signal()
        self.frame_count = 0
        self.fixed_dt: float = 0.016

    def _physics_step(self, dt: float) -> None:
        """Walks all Instances in the World and integrates velocity/gravity."""
        all_inst: list = []
        i: int = 0
        while i < len(self.world.roots):
            self.world._collect_all(self.world.roots[i], all_inst)
            i = i + 1
        j: int = 0
        while j < len(all_inst):
            inst: Instance = all_inst[j]
            inst.physics_step(dt)
            j = j + 1

    def run(self) -> None:
        """Frame order each tick:
          1. refresh input (snapshot keyboard/mouse state)
          2. fire updated(frame) -- scripts move things, read input
          3. physics step -- apply velocity/gravity to all Instances
          4. collision step -- detect AABB overlaps, fire touched()
          5. render
          6. pump window events (returns 0 when window is closed)
        """
        running: int = 1
        while running:
            self.input.refresh()
            self.updated(self.frame_count)
            self._physics_step(self.fixed_dt)
            self.world.step()
            self.world.render()
            running = self.window.update()
            self.frame_count = self.frame_count + 1
        self.window.close()
