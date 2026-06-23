"""Default GLRenderer3D shader pairs, authored with pugtk's Python shader
DSL (_glsl_dsl.py) instead of raw GLSL strings.

Like _glsl_dsl.py itself, this module runs as ordinary CPython at build
time (via the `ast`/`inspect`-based transpile() call below) to produce
plain GLSL text -- the DEFAULT_*_SRC constants are computed once, at
import time, and that's the only thing GLRenderer3D actually ships to
glShaderSource()/glCompileShader(); there's no shader DSL interpretation
happening in the compiled (asmpython) program itself.

Two shader pairs:

- DEFAULT_VERTEX_SRC/DEFAULT_FRAGMENT_SRC: the main-pass shader. Per-
  vertex inputs match GLRenderer3D's interleaved vertex buffer layout
  (see _renderer3d_gl.py's _upload_mesh): position, normal, a flat
  per-vertex base color (every vertex of a given triangle gets that
  triangle's SceneObject.colors[i] -- the GPU equivalent of the software
  renderer's one-shader-call-per-triangle, since attribute interpolation
  would blend each triangle's color into its neighbors' otherwise), and a
  UV coordinate (Mesh's per-triangle-corner tri_u0/v0/u1/v1/u2/v2).
  base_color is the vertex color when useTexture is 0.0, or diffuseTex
  sampled at vUV when useTexture is 1.0 (mix()'d so the GLSL DSL never
  needs a uniform-dependent branch) -- GLRenderer3D sets useTexture per
  draw call based on whether the SceneObject has a Texture.
  Lighting is one shadowed directional light (Blinn-Phong, gamma-
  corrected, PCF-filtered shadow from SHADOW_VERTEX_SRC/
  SHADOW_FRAGMENT_SRC's depth map) plus up to MAX_POINT_LIGHTS unshadowed
  local lights (point or spot -- a spot is a point light with a cone
  cutoff; lightSpotCutoff[i] <= -1.0 means "omnidirectional point light",
  the array-uniform DSL has no separate point/spot type so one array
  covers both). Local lights don't cast shadows (yet) -- shadowing N
  local lights needs either N more shadow maps or a single-pass technique
  (cube/clustered shadow maps), out of scope for this pass; the
  directional light's shadow is the one most scenes need first (sun/
  moon), with local lights filling in fixed unshadowed fill/accent
  lighting (room lamps, torches, etc.).

- SHADOW_VERTEX_SRC/SHADOW_FRAGMENT_SRC: the shadow pass, rendering the
  same geometry from the light's point of view into a depth-only texture
  ahead of the main pass -- see GLRenderer3D._render_shadow_pass(). The
  fragment shader writes nothing (gl_FragDepth is implicit); only depth
  testing/writing is enabled for this pass, with the framebuffer's color
  attachment unused entirely (GL_NONE draw/read buffers).
"""
from __future__ import annotations

from ._glsl_dsl import (
    vertex_shader, fragment_shader,
    vec2, vec3, vec4, mat4, sampler2D,
    Uniform, Varying, set_varying,
    transpile,
)

# Fixed-size local light array -- the GLSL DSL only supports array
# uniforms of a compile-time-literal size (no dynamic SSBOs/UBOs), so
# this is baked into the shader text at transpile time. GLRenderer3D's
# LIGHT counterpart (max_lights) must match.
MAX_POINT_LIGHTS = 4


@vertex_shader
def _default_vs(aPos: vec3, aNormal: vec3, aColor: vec3, aUV: vec2) -> vec4:
    model: mat4 = Uniform("model")
    view_proj: mat4 = Uniform("viewProj")
    light_view_proj: mat4 = Uniform("lightViewProj")
    set_varying("vNormal", aNormal)
    set_varying("vColor", aColor)
    set_varying("vUV", aUV)
    world_pos: vec4 = model * vec4(aPos, 1.0)
    set_varying("vWorldPos", world_pos.xyz)
    light_space_pos: vec4 = light_view_proj * world_pos
    set_varying("vLightSpacePos", light_space_pos.xyz)
    return view_proj * world_pos


@fragment_shader
def _default_fs() -> vec4:
    normal: vec3 = Varying("vNormal")
    vertex_color: vec3 = Varying("vColor")
    uv: vec2 = Varying("vUV")
    world_pos: vec3 = Varying("vWorldPos")
    light_space_pos: vec3 = Varying("vLightSpacePos")
    light_dir: vec3 = Uniform("lightDir")
    ambient: float = Uniform("ambient")
    view_pos: vec3 = Uniform("viewPos")
    shininess: float = Uniform("shininess")
    spec_strength: float = Uniform("specStrength")
    shadow_map: sampler2D = Uniform("shadowMap")
    texel_size: float = Uniform("shadowTexelSize")
    diffuse_tex: sampler2D = Uniform("diffuseTex")
    use_texture: float = Uniform("useTexture")

    point_light_pos: vec3[4] = Uniform("pointLightPos")
    point_light_color: vec3[4] = Uniform("pointLightColor")
    point_light_range: float[4] = Uniform("pointLightRange")
    point_light_spot_dir: vec3[4] = Uniform("pointLightSpotDir")
    point_light_spot_cutoff: float[4] = Uniform("pointLightSpotCutoff")
    point_light_count: int = Uniform("pointLightCount")

    tex_sample: vec3 = texture(diffuse_tex, uv).xyz
    base_color: vec3 = mix(vertex_color, tex_sample, use_texture)

    n: vec3 = normalize(normal)
    view_dir: vec3 = normalize(view_pos - world_pos)
    half_dir: vec3 = normalize(light_dir + view_dir)

    diffuse: float = max(dot(n, light_dir), 0.0)
    spec_angle: float = max(dot(n, half_dir), 0.0)
    specular: float = pow(spec_angle, shininess) * spec_strength

    proj_coords: vec3 = light_space_pos * 0.5 + vec3(0.5, 0.5, 0.5)
    bias: float = max(0.0015 * (1.0 - dot(n, light_dir)), 0.0003)
    current_depth: float = proj_coords.z

    shadow: float = 0.0
    d00: float = texture(shadow_map, proj_coords.xy + vec2(-texel_size, -texel_size)).x
    shadow = shadow + step(d00 + bias, current_depth)
    d01: float = texture(shadow_map, proj_coords.xy + vec2(0.0, -texel_size)).x
    shadow = shadow + step(d01 + bias, current_depth)
    d02: float = texture(shadow_map, proj_coords.xy + vec2(texel_size, -texel_size)).x
    shadow = shadow + step(d02 + bias, current_depth)
    d10: float = texture(shadow_map, proj_coords.xy + vec2(-texel_size, 0.0)).x
    shadow = shadow + step(d10 + bias, current_depth)
    d11: float = texture(shadow_map, proj_coords.xy).x
    shadow = shadow + step(d11 + bias, current_depth)
    d12: float = texture(shadow_map, proj_coords.xy + vec2(texel_size, 0.0)).x
    shadow = shadow + step(d12 + bias, current_depth)
    d20: float = texture(shadow_map, proj_coords.xy + vec2(-texel_size, texel_size)).x
    shadow = shadow + step(d20 + bias, current_depth)
    d21: float = texture(shadow_map, proj_coords.xy + vec2(0.0, texel_size)).x
    shadow = shadow + step(d21 + bias, current_depth)
    d22: float = texture(shadow_map, proj_coords.xy + vec2(texel_size, texel_size)).x
    shadow = shadow + step(d22 + bias, current_depth)
    shadow = shadow / 9.0

    in_range: float = step(0.0, proj_coords.x) * step(proj_coords.x, 1.0) * step(0.0, proj_coords.y) * step(proj_coords.y, 1.0) * step(0.0, proj_coords.z) * step(proj_coords.z, 1.0)
    shadow = shadow * in_range

    light_factor: float = 1.0 - shadow * (1.0 - ambient)
    lit: vec3 = base_color * (ambient + (1.0 - ambient) * diffuse * light_factor) + vec3(1.0, 1.0, 1.0) * specular * light_factor

    local_light_total: vec3 = vec3(0.0, 0.0, 0.0)
    for i in range(4):
        if i < point_light_count:
            to_light: vec3 = point_light_pos[i] - world_pos
            dist: float = length(to_light)
            light_n: vec3 = to_light / dist
            falloff: float = clamp(1.0 - dist / point_light_range[i], 0.0, 1.0)
            atten: float = falloff * falloff

            cutoff: float = point_light_spot_cutoff[i]
            spot_factor: float = 1.0
            if cutoff > -1.0:
                spot_dir_n: vec3 = normalize(point_light_spot_dir[i])
                spot_cos: float = dot(-light_n, spot_dir_n)
                spot_factor = clamp((spot_cos - cutoff) / max(1.0 - cutoff, 0.001), 0.0, 1.0)

            local_diffuse: float = max(dot(n, light_n), 0.0)
            local_half_dir: vec3 = normalize(light_n + view_dir)
            local_spec_angle: float = max(dot(n, local_half_dir), 0.0)
            local_specular: float = pow(local_spec_angle, shininess) * spec_strength
            contribution: float = atten * spot_factor
            local_light_total = local_light_total + point_light_color[i] * (base_color * local_diffuse + vec3(1.0, 1.0, 1.0) * local_specular) * contribution

    lit = lit + local_light_total
    gamma_corrected: vec3 = pow(lit, vec3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
    return vec4(gamma_corrected, 1.0)


@vertex_shader
def _shadow_vs(aPos: vec3, aNormal: vec3, aColor: vec3, aUV: vec2) -> vec4:
    model: mat4 = Uniform("model")
    light_view_proj: mat4 = Uniform("lightViewProj")
    world_pos: vec4 = model * vec4(aPos, 1.0)
    return light_view_proj * world_pos


@fragment_shader
def _shadow_fs() -> vec4:
    # Depth-only pass -- gl_FragDepth is written implicitly from
    # gl_Position.z by fixed-function rasterization; this shader's color
    # output is never read (the shadow-pass framebuffer has no color
    # attachment, see _render_shadow_pass's GL_NONE draw/read buffers).
    return vec4(0.0, 0.0, 0.0, 1.0)


DEFAULT_VERTEX_SRC: str = transpile(_default_vs)
DEFAULT_FRAGMENT_SRC: str = transpile(_default_fs)
SHADOW_VERTEX_SRC: str = transpile(_shadow_vs)
SHADOW_FRAGMENT_SRC: str = transpile(_shadow_fs)
