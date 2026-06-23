"""A small Python-to-GLSL transpiler for authoring GL shaders without
writing raw GLSL strings.

This is a *build-time* tool: it runs as ordinary CPython (using the
`ast`/`inspect` modules) when pugtk itself is being developed/run under
CPython to construct the GLSL source text handed to
lumen.gl.compile_shader(); it is not compiled by asmpython, and the
generated GLSL text is what actually ships to the GPU compiler. There is
no runtime Python-shader interpretation -- by the time a pugtk program is
running (whether under CPython or compiled to native code), every shader
is already a plain GLSL string, going through the exact same
`gl_shader_source_1`/`glCompileShader` pipeline a hand-written GLSL string
would.

Deliberately constrained rather than a general Python-to-anything
compiler: only the vocabulary real vertex/fragment shaders actually need
is supported (typed vec2/vec3/vec4/mat4/float/int locals and parameters,
arithmetic, comparisons, swizzles, a fixed whitelist of GLSL built-in
functions, if/else, and `for i in range(n)` with a literal/constant n).
Anything outside that vocabulary raises ShaderDSLError at transpile time
-- a clear compile-time failure instead of silently wrong or partial GLSL.

Quick start::

    from pugtk._glsl_dsl import vertex_shader, fragment_shader, vec3, vec4, mat4, Uniform, transpile

    @vertex_shader
    def vs(aPos: vec3, aNormal: vec3) -> vec4:
        mvp: mat4 = Uniform("mvp")
        return mvp * vec4(aPos, 1.0)

    @fragment_shader
    def fs(normal: vec3) -> vec4:
        light_dir: vec3 = Uniform("light_dir")
        intensity: float = max(dot(normalize(normal), light_dir), 0.0)
        return vec4(intensity, intensity, intensity, 1.0)

    vertex_glsl = transpile(vs)
    fragment_glsl = transpile(fs)
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field


class ShaderDSLError(Exception):
    """Raised when a decorated shader function uses a Python construct
    outside this DSL's supported vocabulary."""


# ---------------------------------------------------------------------------
# Type markers -- these exist only so shader functions type-check nicely
# under a normal Python linter/IDE and so the transpiler can read
# parameter/return annotations via `inspect.signature`. They carry no
# runtime behavior; a shader function is never actually called from Python.
# ---------------------------------------------------------------------------

class _GLSLType:
    glsl_name: str = ""

    def __init__(self, *args):
        raise ShaderDSLError(
            f"{type(self).__name__}(...) called directly -- shader functions "
            "are transpiled to GLSL, never executed as Python. This "
            "constructor call should only appear inside a @vertex_shader/"
            "@fragment_shader-decorated function, where the transpiler "
            "rewrites it to a GLSL constructor call instead of running it."
        )


class vec2(_GLSLType):
    glsl_name = "vec2"


class vec3(_GLSLType):
    glsl_name = "vec3"


class vec4(_GLSLType):
    glsl_name = "vec4"


class mat3(_GLSLType):
    glsl_name = "mat3"


class mat4(_GLSLType):
    glsl_name = "mat4"


_TYPE_NAMES = {"vec2", "vec3", "vec4", "mat3", "mat4", "float", "int", "bool"}

# GLSL built-in functions this DSL recognizes and passes through verbatim
# (same name, same arity rules as GLSL itself). Calling anything not in
# this set (and not a _GLSLType constructor) raises ShaderDSLError.
_BUILTIN_FUNCS = {
    "dot", "cross", "normalize", "length", "distance", "reflect", "refract",
    "mix", "clamp", "step", "smoothstep", "pow", "exp", "exp2", "log", "log2",
    "sqrt", "inversesqrt", "abs", "sign", "floor", "ceil", "fract", "mod",
    "min", "max", "sin", "cos", "tan", "asin", "acos", "atan",
    "texture", "transpose", "inverse", "determinant",
}

_SWIZZLE_CHARS = set("xyzwrgbastpq")


def Uniform(name: str):
    """Marker for `local: T = Uniform("name")` -- declares a GLSL `uniform`
    of type T (taken from the annotation) bound to `name`, rather than an
    ordinary local variable. Like the _GLSLType classes, never actually
    called; the transpiler pattern-matches this call shape directly."""
    raise ShaderDSLError(
        "Uniform(...) called directly -- only valid as the RHS of an "
        "annotated assignment inside a @vertex_shader/@fragment_shader "
        "function, e.g. `mvp: mat4 = Uniform(\"mvp\")`."
    )


def Varying(name: str):
    """Marker for `local: T = Varying("name")` in a fragment shader --
    declares a GLSL `in` varying of type T received from the vertex
    shader's matching `out`, rather than an ordinary local variable."""
    raise ShaderDSLError(
        "Varying(...) called directly -- only valid as the RHS of an "
        "annotated assignment inside a @fragment_shader function."
    )


@dataclass
class _ShaderFunc:
    """Captured metadata for a decorated shader function, filled in by
    @vertex_shader/@fragment_shader. Holds the parsed AST (not the
    compiled bytecode) so transpile() can walk it."""
    py_func: object
    kind: str  # "vertex" or "fragment"
    tree: ast.FunctionDef = field(default=None)
    source: str = ""


def vertex_shader(fn):
    """Decorator marking a plain Python function as a GLSL vertex shader's
    `main()` body. Parameters are GLSL `in` attributes (location order ==
    parameter order); the return value is gl_Position."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src).body[0]
    if not isinstance(tree, ast.FunctionDef):
        raise ShaderDSLError(f"{fn.__name__} is not a function definition")
    return _ShaderFunc(py_func=fn, kind="vertex", tree=tree, source=src)


def fragment_shader(fn):
    """Decorator marking a plain Python function as a GLSL fragment
    shader's `main()` body. Parameters are GLSL `in` varyings (matched by
    name to the vertex shader's `out` declarations); the return value is
    the fragment color (gl_FragColor-equivalent `out vec4` target)."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src).body[0]
    if not isinstance(tree, ast.FunctionDef):
        raise ShaderDSLError(f"{fn.__name__} is not a function definition")
    return _ShaderFunc(py_func=fn, kind="fragment", tree=tree, source=src)


def _annotation_to_glsl(node: ast.expr | None, ctx: str) -> str:
    if node is None:
        raise ShaderDSLError(f"{ctx}: missing type annotation")
    if isinstance(node, ast.Name) and node.id in _TYPE_NAMES:
        return node.id
    raise ShaderDSLError(f"{ctx}: unsupported type annotation {ast.dump(node)!r}")


class _Transpiler:
    """Walks one shader function's AST, emitting GLSL statement/expression
    text. One instance per transpile() call -- not reused across shaders."""

    def __init__(self, sf: _ShaderFunc):
        self.sf = sf
        self.uniforms: dict[str, str] = {}   # name -> glsl type
        self.varyings_in: dict[str, str] = {}  # name -> glsl type (fragment only)
        self.lines: list[str] = []
        self.indent = 1

    def _emit(self, text: str) -> None:
        self.lines.append(("    " * self.indent) + text)

    def transpile_function(self) -> str:
        fn = self.sf.tree
        body_glsl: list[str] = []
        for stmt in fn.body:
            body_glsl.append(self._stmt(stmt))
        return "\n".join(self.lines)

    # ---- statements -------------------------------------------------------

    def _stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.AnnAssign):
            self._ann_assign(node)
        elif isinstance(node, ast.Assign):
            self._assign(node)
        elif isinstance(node, ast.AugAssign):
            target = self._expr(node.target)
            op = self._binop_str(node.op)
            value = self._expr(node.value)
            self._emit(f"{target} {op}= {value};")
        elif isinstance(node, ast.Return):
            if node.value is None:
                raise ShaderDSLError("return with no value is not supported")
            self._emit_return(node.value)
        elif isinstance(node, ast.If):
            self._if(node)
        elif isinstance(node, ast.For):
            self._for(node)
        elif isinstance(node, ast.Expr):
            # Expression statement (e.g. a bare call) -- emit as a
            # statement with no assignment.
            self._emit(f"{self._expr(node.value)};")
        elif isinstance(node, ast.Pass):
            pass
        else:
            raise ShaderDSLError(f"unsupported statement: {ast.dump(node)}")

    def _emit_return(self, value_node: ast.expr) -> None:
        # Subclass hook: vertex shaders write gl_Position, fragment shaders
        # write the out color. Set by transpile() before calling
        # transpile_function().
        raise NotImplementedError

    def _ann_assign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise ShaderDSLError("annotated assignment target must be a plain name")
        name = node.target.id
        ty = _annotation_to_glsl(node.annotation, f"local {name!r}")
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Uniform"
        ):
            uname = self._literal_str_arg(value, "Uniform")
            self.uniforms[uname] = ty
            return  # uniform declarations are emitted at file level, not inline
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Varying"
        ):
            uname = self._literal_str_arg(value, "Varying")
            self.varyings_in[uname] = ty
            return
        if value is None:
            self._emit(f"{ty} {name};")
        else:
            self._emit(f"{ty} {name} = {self._expr(value)};")

    def _literal_str_arg(self, call: ast.Call, fname: str) -> str:
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise ShaderDSLError(f"{fname}(...) requires exactly one string-literal argument")
        return call.args[0].value

    def _assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise ShaderDSLError("multi-target assignment is not supported")
        target = self._expr(node.targets[0])
        self._emit(f"{target} = {self._expr(node.value)};")

    def _if(self, node: ast.If) -> None:
        self._emit(f"if ({self._expr(node.test)}) {{")
        self.indent += 1
        for s in node.body:
            self._stmt(s)
        self.indent -= 1
        if node.orelse:
            self._emit("} else {")
            self.indent += 1
            for s in node.orelse:
                self._stmt(s)
            self.indent -= 1
        self._emit("}")

    def _for(self, node: ast.For) -> None:
        # Only `for i in range(n)` / `range(a, b)` with int-literal bounds.
        if (
            not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
            or not isinstance(node.iter.func, ast.Name)
            or node.iter.func.id != "range"
        ):
            raise ShaderDSLError(
                "only `for i in range(n)` / `range(a, b)` loops are supported"
            )
        args = node.iter.args
        if len(args) == 1:
            lo, hi = "0", self._expr(args[0])
        elif len(args) == 2:
            lo, hi = self._expr(args[0]), self._expr(args[1])
        else:
            raise ShaderDSLError("range() with a step argument is not supported")
        i = node.target.id
        self._emit(f"for (int {i} = {lo}; {i} < {hi}; {i}++) {{")
        self.indent += 1
        for s in node.body:
            self._stmt(s)
        self.indent -= 1
        self._emit("}")

    # ---- expressions --------------------------------------------------------

    def _expr(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            if isinstance(node.value, (int, float)):
                return self._number_literal(node.value)
            raise ShaderDSLError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.Attribute):
            base = self._expr(node.value)
            if all(c in _SWIZZLE_CHARS for c in node.attr) and 1 <= len(node.attr) <= 4:
                return f"{base}.{node.attr}"
            raise ShaderDSLError(f"unsupported attribute access: .{node.attr}")
        if isinstance(node, ast.BinOp):
            op = self._binop_str(node.op)
            return f"({self._expr(node.left)} {op} {self._expr(node.right)})"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return f"(-{self._expr(node.operand)})"
            if isinstance(node.op, ast.Not):
                return f"(!{self._expr(node.operand)})"
            raise ShaderDSLError(f"unsupported unary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ShaderDSLError("chained comparisons are not supported")
            op = self._cmpop_str(node.ops[0])
            return f"({self._expr(node.left)} {op} {self._expr(node.comparators[0])})"
        if isinstance(node, ast.BoolOp):
            op = "&&" if isinstance(node.op, ast.And) else "||"
            parts = [self._expr(v) for v in node.values]
            return "(" + f" {op} ".join(parts) + ")"
        if isinstance(node, ast.Call):
            return self._call(node)
        raise ShaderDSLError(f"unsupported expression: {ast.dump(node)}")

    def _call(self, node: ast.Call) -> str:
        if not isinstance(node.func, ast.Name):
            raise ShaderDSLError("only direct name calls are supported (no method calls)")
        fname = node.func.id
        if fname in _TYPE_NAMES or fname in _BUILTIN_FUNCS:
            args = ", ".join(self._expr(a) for a in node.args)
            return f"{fname}({args})"
        raise ShaderDSLError(
            f"call to {fname!r} is not supported -- only GLSL constructors "
            f"({sorted(_TYPE_NAMES)}) and built-in functions "
            f"({sorted(_BUILTIN_FUNCS)}) may be called"
        )

    def _number_literal(self, v) -> str:
        if isinstance(v, int):
            return str(v)
        return repr(float(v))

    def _binop_str(self, op: ast.operator) -> str:
        m = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.Mod: "%",
        }
        for k, v in m.items():
            if isinstance(op, k):
                return v
        raise ShaderDSLError(f"unsupported binary operator: {ast.dump(op)}")

    def _cmpop_str(self, op: ast.cmpop) -> str:
        m = {
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!=",
        }
        for k, v in m.items():
            if isinstance(op, k):
                return v
        raise ShaderDSLError(f"unsupported comparison operator: {ast.dump(op)}")


class _VertexTranspiler(_Transpiler):
    def _emit_return(self, value_node: ast.expr) -> None:
        self._emit(f"gl_Position = {self._expr(value_node)};")


class _FragmentTranspiler(_Transpiler):
    OUT_NAME = "fragColor"

    def _emit_return(self, value_node: ast.expr) -> None:
        self._emit(f"{self.OUT_NAME} = {self._expr(value_node)};")


def transpile(sf: _ShaderFunc, version: str = "330 core") -> str:
    """Transpile a @vertex_shader/@fragment_shader-decorated function into
    a complete GLSL source string, ready for gl_shader_source_1()/
    glCompileShader()."""
    fn = sf.tree
    params: list[tuple[str, str]] = []
    for a in fn.args.args:
        params.append((a.arg, _annotation_to_glsl(a.annotation, f"parameter {a.arg!r}")))
    ret_ty = _annotation_to_glsl(fn.returns, "return type")
    if ret_ty != "vec4":
        raise ShaderDSLError(
            f"{sf.kind} shader must return vec4 "
            f"({'gl_Position' if sf.kind == 'vertex' else 'fragment color'}), got {ret_ty!r}"
        )

    if sf.kind == "vertex":
        t = _VertexTranspiler(sf)
    else:
        t = _FragmentTranspiler(sf)
    body = t.transpile_function()

    lines: list[str] = [f"#version {version}"]
    if sf.kind == "vertex":
        for i, (pname, pty) in enumerate(params):
            lines.append(f"layout(location = {i}) in {pty} {pname};")
        for vname, vty in t.varyings_in.items():
            lines.append(f"out {vty} {vname};")
    else:
        for pname, pty in params:
            lines.append(f"in {pty} {pname};")
        for vname, vty in t.varyings_in.items():
            lines.append(f"in {vty} {vname};")
        lines.append(f"out vec4 {_FragmentTranspiler.OUT_NAME};")

    for uname, uty in t.uniforms.items():
        lines.append(f"uniform {uty} {uname};")

    lines.append("void main() {")
    lines.append(body)
    lines.append("}")
    return "\n".join(lines)
