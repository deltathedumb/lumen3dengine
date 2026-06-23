"""Pre-transpiled GLSL output of _gl_shaders.py's Python shader DSL
source, as plain string constants -- this is the file _renderer3d_gl.py
actually imports.

_gl_shaders.py itself uses Python's `ast`/`inspect` modules (via
_glsl_dsl.transpile()) to turn @vertex_shader/@fragment_shader-decorated
functions into GLSL text; those modules aren't part of asmpython's
stdlib, so _gl_shaders.py can only run under plain CPython, never inside
a program asmpython compiles. This file is the bridge: regenerate it by
running _gl_shaders.py under CPython (see its own docstring) and pasting
the resulting DEFAULT_VERTEX_SRC/DEFAULT_FRAGMENT_SRC values back in here
whenever the DSL source shaders in _gl_shaders.py change. Everything
downstream (_renderer3d_gl.py, and any program asmpython compiles) only
ever sees the plain strings below.
"""
from __future__ import annotations

DEFAULT_VERTEX_SRC: str = '#version 330 core\nlayout(location = 0) in vec3 aPos;\nlayout(location = 1) in vec3 aNormal;\nlayout(location = 2) in vec3 aColor;\nout vec3 vNormal;\nout vec3 vColor;\nout vec3 vWorldPos;\nuniform mat4 model;\nuniform mat4 viewProj;\nvoid main() {\n    vNormal = aNormal;\n    vColor = aColor;\n    vec4 world_pos = (model * vec4(aPos, 1.0));\n    vWorldPos = world_pos.xyz;\n    gl_Position = (viewProj * world_pos);\n}'
DEFAULT_FRAGMENT_SRC: str = '#version 330 core\nin vec3 vNormal;\nin vec3 vColor;\nout vec4 fragColor;\nuniform vec3 lightDir;\nuniform float ambient;\nvoid main() {\n    vec3 n = normalize(vNormal);\n    float intensity = max(dot(n, lightDir), 0.0);\n    float factor = (ambient + ((1.0 - ambient) * intensity));\n    fragColor = vec4((vColor * factor), 1.0);\n}'
