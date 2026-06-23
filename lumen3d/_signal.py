"""Signal -- a Roblox-style event/callback object: `.connect(fn)` to
register a handler, call the Signal itself to fire it.

This is lumen3dengine's scripting layer, not pugtk (pugtk is graphics-
only -- see pugtk's own module docstrings). Built on two real asmpython
compiler features verified working this session:

- Calling a value stored in an attribute (`part.touched(value)`, where
  `touched` is a Signal instance field, not a literal method) -- sema
  rewrites `obj.field(args)` into "read the field, call its __call__"
  when `field` resolves to an instance-typed class field whose class has
  `__call__`, reusing the existing instance-method dispatch path.
- A function reference stored in a list and called back later
  (`self.handlers[i]` read into a local, then called) -- already worked,
  but calling it from inside an arithmetic expression (`total = total +
  fn(value)`) was a genuine compiler bug (raw push/pop around a call
  broke 16-byte stack alignment whenever the called function's own body
  contained a further call) fixed this session.
"""
from __future__ import annotations


class Signal:
    handlers: list

    def __init__(self) -> None:
        self.handlers = []

    def connect(self, fn) -> None:
        """Register a handler. `fn` may be a plain function or a
        closure; called with whatever arguments `fire()`/`__call__()`
        is invoked with."""
        self.handlers.append(fn)

    def fire(self, value: int) -> int:
        """Calls every connected handler with `value`, in connection
        order. Returns the number of handlers invoked. A single `int`
        argument (rather than *args) since asmpython's static type
        system has no variadic-and-polymorphic call signature -- a
        scene/script layer wanting to pass richer data should pass an
        Instance/struct-shaped value instead of relying on multiple
        positional args."""
        count: int = 0
        i: int = 0
        while i < len(self.handlers):
            fn = self.handlers[i]
            fn(value)
            count = count + 1
            i = i + 1
        return count

    def __call__(self, value: int) -> int:
        """`signal(value)` is sugar for `signal.fire(value)` -- this is
        what makes `part.touched(other)` read like firing an event,
        matching Roblox's `part.Touched:Fire(other)` / the implicit fire
        on collision."""
        return self.fire(value)
