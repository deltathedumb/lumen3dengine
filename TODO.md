# Deferred work

Things identified during development, deliberately not fixed/built yet,
with enough context to pick back up later.

## Compiler (asmpython)

### `__setattr__`/`__getattr__` interception — not implemented, deferred

Investigated for the scriptability push (lumen3dengine's Roblox-comparison
goal). Real interception (`self.x = value` automatically routing through a
user-defined hook before the normal dict-set) is **not supported at all**
today — `ast_nodes.py`/`sema.py`/`codegen.py` have zero references to
`__setattr__`/`__getattr__`; attribute access always compiles straight to
`_runtime_dict_get_default`/`_runtime_dict_set`, no hook point exists.

Decided to skip it: `@property`/`@x.setter` already gives the same
call-site ergonomics (`part.x = value` transparently calls a setter) with
**zero compiler risk**, confirmed working via a real test (Signal fired
from inside a `@x.setter`). The gap vs true `__setattr__` is only that
each reactive field needs one `@property`/`@x.setter` pair written
explicitly, instead of every bare attribute write being automatically
interceptable with no per-field opt-in. Given two unrelated real bugs were
already found and fixed in the same session (see below), this was
correctly judged not worth the added risk for the ergonomics gap it closes.

If revisited: the hook point would need to live in `codegen.py`'s
`AttrAssign` handling (~line 2871) and the `Attr` read path, checking
`cls.methods` for `__setattr__`/`__getattr__` before falling through to
the direct dict op — mirroring how `__call__` dispatch already works
(`sema.py` ~line 7118-7149, `codegen.py` ~line 13512-13522).

### `open()` builtin — confirmed broken, not fixed

`open(path, mode)` is recognized by `sema.py` (arity table + `"any"`
return type) but has **zero codegen implementation** — any call falls
through to the generic "unknown external function, evaluate args, return
0" stub (`codegen.py` ~line 13616). Any `.read()`/`.write()`/`.close()`
on the result similarly falls through to the generic "unknown method on
an `any`-typed receiver" stub (`codegen.py` ~line 11476). Net effect:
`f = open(path); f.read()` compiles cleanly and segfaults at runtime
(calls `strlen` on a NULL pointer).

**Workaround in use today**: `io.FileIO`/`io.TextIOWrapper` (real working
classes built on `os.fopen`/`fgetc`/`fputs`/`fclose`, already used by both
mesh loaders) work correctly when constructed directly —
`from io import FileIO; f = FileIO(path, "r")` — bypassing the broken
builtin entirely. No code should call bare `open()`.

If revisited: cleanest fix is rewriting the `open(...)` call in sema into
a constructor call for `io.FileIO`/`io.TextIOWrapper` (so the existing,
working `instance:`-prefixed method-dispatch path picks up `.read()` etc.
for free) rather than adding new codegen branches — see investigation
notes from the asset-loading session for the full writeup of why this is
a small, contained gap (the underlying FFI/class infrastructure already
works) and not a from-scratch feature.

### `AttrAssign` codegen uses raw push/pop — latent same-class bug as the BinOp fix, unconfirmed

`codegen.py`'s `A.AttrAssign` handling (~line 2884-2895, `obj.name =
value`) does:
```
self.gen_expr(stmt.value, info)
self.emitf("push rax")
self.gen_expr(stmt.obj, info)   # <- may itself contain a call
self.emitf("pop rcx", ...)
```
This is the same `push`/`pop`-across-a-possible-call pattern that was
confirmed to break 16-byte stack alignment and segfault for `A.BinOp`'s
plain-int fallback this session (see git history / asmpython CHANGELOG
for the `total = total + callee()` fix). Not yet confirmed to actually
manifest here — `stmt.obj` (the attribute's receiver) evaluating to
something that itself contains a call (e.g. `get_part().x = value`) is
the trigger shape, not yet tested. Same fix pattern applies: spill
`value` into a dedicated `info.locals_[f"__attrassign_{id(stmt)}"]` frame
slot instead of `push rax`/`pop rcx`.

### `pow(base, exp)` codegen uses raw push/pop — same latent bug class, unconfirmed

`codegen.py` ~line 13494-13498: `push rax` around evaluating the exponent
argument. If the exponent expression itself contains a call (e.g.
`pow(2, compute_exponent())`), same alignment hazard as above. Not yet
tested. Same fix: frame slot instead of push/pop.

## Build / installation

### Installed asmpython vs local repo — must keep in sync

**CRITICAL**: `python -m asmpython` uses the *installed* package at
`site-packages/asmpython`, NOT the local repo at
`lumen3d/asmpython/asmpython`. Any compiler edits to the local repo
are silently ignored by the compiler until the package is reinstalled.

After every compiler edit, sync to the installed location:
```
cp -r asmpython/asmpython/_compiler/* \
  /c/Users/jessj/AppData/Local/Programs/Python/Python312/Lib/site-packages/asmpython/_compiler/
cp -r asmpython/asmpython/stdlib/* \
  /c/Users/jessj/AppData/Local/Programs/Python/Python312/Lib/site-packages/asmpython/stdlib/
```
Or install editable: `pip install -e asmpython/`

Confirmed the fix was found in session 2026-06-23: the callable-field
dispatch sema fix (Signal.fire via `part.touched(x)`) and BinOp
frame-slot fix were both applied only to the local repo and had zero
effect until the installed package was overwritten.

## Engine layer

### Scope note: `pugtk` is graphics-only

Confirmed by the user mid-session: `pugtk` (c:\Users\jessj\Documents\coding\lumen3d\pugtk)
is the **graphics layer only**. Scriptability (Signal/event system,
persistent game-object tree, main-loop abstraction) belongs in a
**separate engine layer above pugtk**, not bolted onto pugtk itself.
pugtk should only ever be *called into* by that layer for rendering.
