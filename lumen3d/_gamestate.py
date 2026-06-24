"""GameState -- simple state machine for game screens.

Replaces ad-hoc `if mode == "playing"` checks scattered through scripts
with a proper push/pop state stack used in every commercial engine.

Usage:
    class PlayingState(GameState):
        def on_enter(self) -> None:
            print("entered playing")
        def on_update(self, dt: float, frame: int) -> None:
            # move player, check win condition ...
            pass
        def on_exit(self) -> None:
            print("left playing")

    sm = StateMachine()
    sm.push(PlayingState())
    sm.update(dt, frame)   # call from GameLoop updated callback

The state machine integrates with GameLoop via StateMachine.connect(loop):
    sm.connect(loop)  # auto-registers update callback

States can push/pop other states through the machine reference set by
the machine itself on enter:
    self.machine.push(PauseState())
    self.machine.pop()
"""
from __future__ import annotations


class GameState:
    """Base class for a single screen / mode of the game.

    Override on_enter, on_update, on_exit. The machine field is set
    automatically before on_enter is called.
    """

    machine: "StateMachine"

    def __init__(self) -> None:
        self.machine = None

    def on_enter(self) -> None:
        pass

    def on_update(self, dt: float, frame: int) -> None:
        pass

    def on_exit(self) -> None:
        pass

    def on_pause(self) -> None:
        """Called when a new state is pushed on top of this one."""
        pass

    def on_resume(self) -> None:
        """Called when the state above this one is popped."""
        pass


class StateMachine:
    """Push/pop stack of GameStates. The top state receives update calls.

    connect(loop) wires update into GameLoop.updated automatically.
    """

    _stack: list
    _fixed_dt: float
    _loop_ref: object

    def __init__(self) -> None:
        self._stack = []
        self._fixed_dt = 0.016
        self._loop_ref = None

    def push(self, state: GameState) -> None:
        if len(self._stack) > 0:
            self._stack[len(self._stack) - 1].on_pause()
        state.machine = self
        self._stack.append(state)
        state.on_enter()

    def pop(self) -> None:
        if len(self._stack) == 0:
            return
        top: GameState = self._stack[len(self._stack) - 1]
        top.on_exit()
        new_len: int = len(self._stack) - 1
        self._stack = self._stack[:new_len]
        if len(self._stack) > 0:
            self._stack[len(self._stack) - 1].on_resume()

    def replace(self, state: GameState) -> None:
        """Pop current state and push new one atomically."""
        if len(self._stack) > 0:
            top: GameState = self._stack[len(self._stack) - 1]
            top.on_exit()
            new_len: int = len(self._stack) - 1
            self._stack = self._stack[:new_len]
        state.machine = self
        self._stack.append(state)
        state.on_enter()

    def update(self, dt: float, frame: int) -> None:
        if len(self._stack) == 0:
            return
        self._stack[len(self._stack) - 1].on_update(dt, frame)

    def connect(self, loop) -> None:
        """Wire this state machine into a GameLoop.
        Call once before loop.run()."""
        self._fixed_dt = loop.fixed_dt
        self._loop_ref = loop
        loop.updated.connect(self._loop_update)

    def _loop_update(self, frame: int) -> None:
        self.update(self._fixed_dt, frame)

    def current(self) -> GameState:
        if len(self._stack) == 0:
            return None
        return self._stack[len(self._stack) - 1]

    def depth(self) -> int:
        return len(self._stack)

    def is_empty(self) -> int:
        return 1 if len(self._stack) == 0 else 0
