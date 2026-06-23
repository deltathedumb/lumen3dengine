"""Default GLRenderer3D vertex/fragment shader pair, authored with pugtk's
Python shader DSL (_glsl_dsl.py) instead of raw GLSL strings.

Like _glsl_dsl.py itself, this module runs as ordinary CPython at build
time (via the `ast`/`inspect`-based transpile() call below) to produce
plain GLSL text -- DEFAULT_VERTEX_SRC/DEFAULT_FRAGMENT_SRC are computed
once, at import time, and that's the only thing GLRenderer3D actually
ships to glShaderSource()/glCompileShader(); there's no shader DSL
interpretation happening in the compiled (asmpython) program itself.

Per-vertex inputs match GLRenderer3D's interleaved vertex buffer layout
(see _renderer3d_gl.py's _upload_mesh): position, normal, UV, and a flat
per-vertex base color (every vertex of a given triangle gets that
triangle's SceneObject.colors[i] -- the GPU equivalent of the software
renderer's one-shader-call-per-triangle, since attribute interpolation
would blend each triangle's color into its neighbors' otherwise). Lighting
is plain Lambertian (ambient floor + diffuse), matching pugtk's existing
lambert_shader() in _shading.py -- the well-understood default both
renderer backends share, not a coincidence.
"""
from __future__ import annotations

from ._glsl_dsl import (
    vertex_shader, fragment_shader,
    vec3, vec4, mat4,
    Uniform, Varying, set_varying,
    transpile,
)


@vertex_shader
def _default_vs(aPos: vec3, aNormal: vec3, aColor: vec3) -> vec4:
    model: mat4 = Uniform("model")
    view_proj: mat4 = Uniform("viewProj")
    set_varying("vNormal", aNormal)
    set_varying("vColor", aColor)
    world_pos: vec4 = model * vec4(aPos, 1.0)
    set_varying("vWorldPos", world_pos.xyz)
    return view_proj * world_pos


@fragment_shader
def _default_fs() -> vec4:
    normal: vec3 = Varying("vNormal")
    base_color: vec3 = Varying("vColor")
    light_dir: vec3 = Uniform("lightDir")
    ambient: float = Uniform("ambient")
    n: vec3 = normalize(normal)
    intensity: float = max(dot(n, light_dir), 0.0)
    factor: float = ambient + (1.0 - ambient) * intensity
    return vec4(base_color * factor, 1.0)


DEFAULT_VERTEX_SRC: str = transpile(_default_vs)
DEFAULT_FRAGMENT_SRC: str = transpile(_default_fs)
