"""GLRenderer3D — a GPU (OpenGL) backend for pugtk's Mesh/Camera/
SceneObject API, built on asmpython's gl_import() mechanism.

Same shape of inputs as the software Renderer3D (_renderer3d.py):
construct once with a Window + Camera, then call render_scene(objects)
every frame. Internally, each Mesh is uploaded to a VBO/VAO exactly once
(cached on the Mesh itself -- see _mesh.py's gl_uploaded/gl_vao/gl_vbo
fields) and redrawn via glDrawArrays on every subsequent frame, instead of
the software renderer's per-pixel Python rasterization loop.

Vertex format (interleaved, 9 floats/vertex -- see _upload_mesh):
    position.xyz, normal.xyz, color.rgb
One vertex per triangle corner (3 per triangle, not shared/indexed):
Mesh's per-vertex normal is genuinely shared across triangles (Gouraud-
style averaged normals), but the per-triangle SceneObject.colors[i] is
not, so every vertex needs its own copy of whichever triangle it belongs
to's color regardless -- an indexed buffer would still have to duplicate
position+normal per triangle corner to let color vary per-triangle, so a
flat non-indexed buffer is simplest and isn't meaningfully more memory for
the low-to-mid-poly meshes this engine targets.

Shading is the default Lambertian vertex+fragment pair from
_gl_shaders.py; swap renderer.vertex_src/fragment_src (and call
recompile_shader()) for a custom GLSL pair, e.g. from your own
pugtk._glsl_dsl-authored shader functions.

`@glfns.imported` methods on this class are GLRenderer3D's own GL
bindings -- see asmpython's CHANGELOG for the class-method @imported
mechanism this relies on. `glfns` itself must be a module-level global
(Python evaluates class decorators once, at class-definition time), so it
lives here at import time, shared by every GLRenderer3D instance (GL
function pointers don't vary per-instance -- they're resolved once
against whichever GL context is current when gl_import() ran).
"""
from __future__ import annotations

import lumen.gl as gl
import _gui_sdl

from ._vector import Vector3
from ._matrix import Matrix4
from ._mesh import Mesh
from ._camera import Camera
from ._scene import SceneObject
from ._light import PointLight
from ._texture import Texture
from ._gl_shaders_compiled import (
    DEFAULT_VERTEX_SRC, DEFAULT_FRAGMENT_SRC,
    SHADOW_VERTEX_SRC, SHADOW_FRAGMENT_SRC,
)

glfns = gl_import()


# _pack_floats32(src_buf: list[float], dst_buf: list, count: int) -- raw
# NASM body (@assembly_func bodies must be exactly one string literal of
# pure NASM; this comment carries the explanation a docstring normally
# would). Converts `count` doubles from src_buf's raw backing buffer into
# packed 32-bit floats written into dst_buf's raw backing buffer (2 floats
# per 8-byte list[int] slot, low half then high half -- same packing
# lumen.PixelBuffer already uses for its own 2-values-per-slot layout).
# dst_buf must have at least ceil(count/2) elements.
#
# Exists because glUniformMatrix4fv (and any other GL entry point declared
# GLfloat*) reads packed 32-bit floats, but asmpython's `float` is always
# a 64-bit double and list[float] stores 8 bytes/element -- passing a
# list[float] directly would have the driver read every other 4 bytes as
# a float and the other half as wrong-width garbage. This is the buffer-
# data equivalent of _gen_dynamic_call's scalar-argument cvtsd2ss
# narrowing (see asmpython's CHANGELOG), just done once in pugtk-level
# code instead of needing another compiler special-case for every
# possible GLfloat* parameter.
#
# Win64 ABI only (rcx/rdx/r8 args) -- pugtk's GL backend is exercised
# against --target windows this session; a SysV (rdi/rsi/rdx) sibling
# would be needed before this runs on Linux. LIST_BUF_OFF (16) is the
# list header's backing-buffer-pointer field offset, matching every other
# list_buf access in this codebase.
@assembly_func
def _pack_floats32(src_buf: list, dst_buf: list, count: int) -> int:
    """
    mov r9, rcx             ; src_buf header
    mov r10, rdx            ; dst_buf header
    mov r11, r8              ; count
    mov r9, [r9+16]          ; r9 = src raw double buffer
    mov r10, [r10+16]        ; r10 = dst raw int64 (2x float32) buffer
    xor rax, rax              ; i = 0
.pack_loop:
    cmp rax, r11
    jge .pack_done
    mov rcx, rax
    shl rcx, 3
    movsd xmm0, [r9+rcx]      ; xmm0 = src[i] (double)
    cvtsd2ss xmm0, xmm0       ; narrow to float32
    movd ecx, xmm0             ; ecx = float32 bits
    mov rdx, rax
    shr rdx, 1                 ; slot = i / 2
    mov r8, rax
    and r8, 1                   ; i % 2
    shl rdx, 3                  ; byte offset = slot * 8
    cmp r8, 0
    jne .pack_high
    mov dword [r10+rdx], ecx        ; low 32 bits of the slot
    jmp .pack_next
.pack_high:
    mov dword [r10+rdx+4], ecx      ; high 32 bits of the slot
.pack_next:
    inc rax
    jmp .pack_loop
.pack_done:
    xor rax, rax
    ret
    """


# _pack_rgba8(src_buf: list[int], dst_buf: list, count: int) -- raw NASM
# body. Converts `count` packed-int pixels (Texture's 0x00RRGGBB format,
# one 64-bit asmpython int per pixel) into tightly-packed RGBA8 bytes (R,
# G, B, 255) written into dst_buf's raw backing buffer, 2 pixels per
# 8-byte list[int] slot (8 bytes = 2 pixels * 4 bytes/pixel) -- what
# glTexImage2D(..., GL_RGBA, GL_UNSIGNED_BYTE, ...) actually reads.
# dst_buf must have at least ceil(count/2) elements. Win64 ABI only, same
# rationale as _pack_floats32 above.
@assembly_func
def _pack_rgba8(src_buf: list, dst_buf: list, count: int) -> int:
    """
    mov r9, rcx               ; src_buf header
    mov r10, rdx              ; dst_buf header
    mov r11, r8                ; count
    mov r9, [r9+16]            ; r9 = src raw int64 (packed 0x00RRGGBB) buffer
    mov r10, [r10+16]          ; r10 = dst raw byte buffer
    xor rax, rax                ; i = 0
.rgba_loop:
    cmp rax, r11
    jge .rgba_done
    mov rdx, rax
    shl rdx, 3                  ; src byte offset = i * 8
    mov rdx, [r9+rdx]           ; rdx = src[i] (0x00RRGGBB)
    mov rcx, rax
    shl rcx, 2                  ; dst byte offset = i * 4 (computed before
                                 ; rcx is reused for the green channel below)
    mov r8, rdx
    shr r8, 16
    and r8, 0xFF                 ; r8 = R
    mov byte [r10+rcx], r8b
    mov r8, rdx
    shr r8, 8
    and r8, 0xFF                  ; r8 = G
    mov byte [r10+rcx+1], r8b
    mov r8, rdx
    and r8, 0xFF                   ; r8 = B
    mov byte [r10+rcx+2], r8b
    mov byte [r10+rcx+3], 255
    inc rax
    jmp .rgba_loop
.rgba_done:
    xor rax, rax
    ret
    """


# _gl_tex_image_2d_data(fn_ptr, target, level, internalformat, width,
# height, border, format, ty, data_ptr) -- raw NASM body. glTexImage2D's
# real signature needs a `data` argument that's sometimes NULL (0, when
# only allocating storage -- the existing @glfns.imported glTexImage2D
# stub above, data: int, already covers that) and sometimes a real pixel
# buffer pointer (when actually uploading texture data) -- @imported
# stubs are typed per-parameter with no overloading, so one Python method
# can't serve both call shapes. This hand-rolled version takes
# glTexImage2D's own resolved pointer (via gl_resolve(glfns,
# "glTexImage2D"), same pattern as gl.shader_source()'s
# getattr-free glShaderSource pointer) and calls it directly so `data`
# can be a real list_buf pointer instead. Win64 ABI only, matching every
# other @assembly_func in this file.
#
# Argument layout: this function's own first 4 params (fn_ptr, target,
# level, internalformat) arrive in rcx/rdx/r8/r9; its remaining 6 params
# (width, height, border, format, ty, data_ptr) are stack-passed to it at
# [rsp+40] through [rsp+80] (rsp+0 is the return address, rsp+8..39 is
# the 32-byte shadow space its own caller reserved). Calling the real
# glTexImage2D (9 args) through fn_ptr needs its own shadow space + 5
# stack slots (args 5-9: height, border, format, ty, data_ptr) set up
# below the indirect call.
@assembly_func
def _gl_tex_image_2d_data(
    fn_ptr: int, target: int, level: int, internalformat: int,
    width: int, height: int, border: int, format: int, ty: int, data_ptr: list,
) -> int:
    """
    mov rax, rcx              ; rax = fn_ptr (freed rcx for real-call arg 1)
    mov r10, [rsp+40]          ; width  (this func's 5th param)
    mov r11, [rsp+48]          ; height (6th)
    sub rsp, 88                 ; 32 shadow + 5*8 stack args + 8 align = 88
    mov [rsp+32], r11           ; real call arg5 = height
    mov r11, [rsp+88+56]        ; border (7th param, offset by the sub rsp above)
    mov [rsp+40], r11           ; real call arg6 = border
    mov r11, [rsp+88+64]        ; format (8th param)
    mov [rsp+48], r11           ; real call arg7 = format
    mov r11, [rsp+88+72]        ; ty (9th param)
    mov [rsp+56], r11           ; real call arg8 = ty
    mov r11, [rsp+88+80]        ; data_ptr (10th param) -- a list header
                                 ; pointer (same representation every other
                                 ; list-typed value uses); GL needs the raw
                                 ; backing buffer at +LIST_BUF_OFF (16), not
                                 ; the header itself -- same dereference
                                 ; _pack_floats32/_pack_rgba8 do for their
                                 ; own list-typed params.
    mov r11, [r11+16]
    mov [rsp+64], r11           ; real call arg9 = data_ptr's raw buffer
    mov rcx, rdx                ; real call arg1 = target
    mov rdx, r8                 ; real call arg2 = level
    mov r8, r9                  ; real call arg3 = internalformat
    mov r9, r10                 ; real call arg4 = width
    call rax
    add rsp, 88
    ret
    """


class GLWindow:
    """An OpenGL-capable window -- GLRenderer3D's counterpart to the
    software renderer's Window (_window.py), which wraps a non-GL
    lumen.Canvas/PixelBuffer pair instead. Needs its own window type
    because GL context creation requires the WINDOW_OPENGL flag at window-
    creation time (lumen.gl.create_gl_window), which Window.__init__
    never sets. Same width/height shape as Window so GLRenderer3D reads
    window.width/window.height identically either way.
    """

    width: int
    height: int
    win: int
    ctx: int

    def __init__(self, title: str, width: int, height: int):
        self.width = width
        self.height = height
        self.win = gl.create_gl_window(title, width, height)
        self.ctx = gl.create_context(self.win)

    def update(self) -> int:
        """Present the current frame and pump SDL's event queue, matching
        Window.update()'s call shape -- returns 0 when the user closes the
        window (EVENT_QUIT), 1 otherwise."""
        gl.swap_window(self.win)
        ev: int = _gui_sdl.poll_event()
        while ev != 0:
            if ev == 0x100:  # EVENT_QUIT
                return 0
            ev = _gui_sdl.poll_event()
        return 1

    def close(self):
        gl.destroy(self.win, self.ctx)


class GLRenderer3D:
    window: GLWindow
    camera: Camera
    light_dir: Vector3
    ambient: float
    shininess: float
    spec_strength: float
    vertex_src: str
    fragment_src: str
    shadows_enabled: int
    shadow_extent: float
    shadow_map_size: int
    point_lights: list[PointLight]
    max_lights: int

    program: int
    _u_model: int
    _u_view_proj: int
    _u_light_dir: int
    _u_ambient: int
    _u_view_pos: int
    _u_shininess: int
    _u_spec_strength: int
    _u_light_view_proj: int
    _u_shadow_map: int
    _u_shadow_texel_size: int
    _u_point_light_pos: int
    _u_point_light_color: int
    _u_point_light_range: int
    _u_point_light_spot_dir: int
    _u_point_light_spot_cutoff: int
    _u_point_light_count: int
    _u_diffuse_tex: int
    _u_use_texture: int
    _initialized: int

    shadow_program: int
    _su_model: int
    _su_light_view_proj: int
    _shadow_fbo: int
    _shadow_tex: int

    @glfns.imported
    def glGetError(self) -> int:
        return 0

    @glfns.imported
    def glShaderSource(self) -> int:
        # Never called directly -- glShaderSource's real signature (char**)
        # goes through lumen.gl.shader_source()/_gl_shader_source_1
        # instead. This stub only registers the function pointer.
        return 0

    @glfns.imported
    def glCreateShader(self, shader_type: int) -> int:
        return 0

    @glfns.imported
    def glCompileShader(self, shader: int) -> int:
        return 0

    @glfns.imported
    def glGetShaderiv(self, shader: int, pname: int, params: list) -> int:
        return 0

    @glfns.imported
    def glCreateProgram(self) -> int:
        return 0

    @glfns.imported
    def glAttachShader(self, program: int, shader: int) -> int:
        return 0

    @glfns.imported
    def glLinkProgram(self, program: int) -> int:
        return 0

    @glfns.imported
    def glGetProgramiv(self, program: int, pname: int, params: list) -> int:
        return 0

    @glfns.imported
    def glDeleteShader(self, shader: int) -> int:
        return 0

    @glfns.imported
    def glUseProgram(self, program: int) -> int:
        return 0

    @glfns.imported
    def glGenVertexArrays(self, n: int, arrays: list) -> int:
        return 0

    @glfns.imported
    def glBindVertexArray(self, array: int) -> int:
        return 0

    @glfns.imported
    def glGenBuffers(self, n: int, buffers: list) -> int:
        return 0

    @glfns.imported
    def glBindBuffer(self, target: int, buffer: int) -> int:
        return 0

    @glfns.imported
    def glBufferData(self, target: int, size: int, data: list, usage: int) -> int:
        return 0

    @glfns.imported
    def glVertexAttribPointer(self, index: int, size: int, ty: int, normalized: int, stride: int, offset: int) -> int:
        return 0

    @glfns.imported
    def glEnableVertexAttribArray(self, index: int) -> int:
        return 0

    @glfns.imported
    def glGetUniformLocation(self, program: int, name: str) -> int:
        return 0

    @glfns.imported
    def glUniformMatrix4fv(self, location: int, count: int, transpose: int, value: list) -> int:
        return 0

    @glfns.imported
    def glUniform1f(self, location: int, v0: float) -> int:
        return 0

    @glfns.imported
    def glUniform3f(self, location: int, v0: float, v1: float, v2: float) -> int:
        return 0

    @glfns.imported
    def glUniform3fv(self, location: int, count: int, value: list) -> int:
        return 0

    @glfns.imported
    def glUniform1fv(self, location: int, count: int, value: list) -> int:
        return 0

    @glfns.imported
    def glEnable(self, cap: int) -> int:
        return 0

    @glfns.imported
    def glDepthFunc(self, func: int) -> int:
        return 0

    @glfns.imported
    def glCullFace(self, mode: int) -> int:
        return 0

    @glfns.imported
    def glFrontFace(self, mode: int) -> int:
        return 0

    @glfns.imported
    def glClearColor(self, r: float, g: float, b: float, a: float) -> int:
        return 0

    @glfns.imported
    def glClear(self, mask: int) -> int:
        return 0

    @glfns.imported
    def glViewport(self, x: int, y: int, w: int, h: int) -> int:
        return 0

    @glfns.imported
    def glDrawArrays(self, mode: int, first: int, count: int) -> int:
        return 0

    @glfns.imported
    def glUniform1i(self, location: int, v0: int) -> int:
        return 0

    @glfns.imported
    def glGenFramebuffers(self, n: int, framebuffers: list) -> int:
        return 0

    @glfns.imported
    def glBindFramebuffer(self, target: int, framebuffer: int) -> int:
        return 0

    @glfns.imported
    def glFramebufferTexture2D(self, target: int, attachment: int, textarget: int, texture: int, level: int) -> int:
        return 0

    @glfns.imported
    def glGenTextures(self, n: int, textures: list) -> int:
        return 0

    @glfns.imported
    def glBindTexture(self, target: int, texture: int) -> int:
        return 0

    @glfns.imported
    def glTexImage2D(self, target: int, level: int, internalformat: int, width: int, height: int, border: int, format: int, ty: int, data: int) -> int:
        return 0

    @glfns.imported
    def glTexParameteri(self, target: int, pname: int, param: int) -> int:
        return 0

    @glfns.imported
    def glTexParameterfv(self, target: int, pname: int, params: list) -> int:
        return 0

    @glfns.imported
    def glDrawBuffer(self, mode: int) -> int:
        return 0

    @glfns.imported
    def glReadBuffer(self, mode: int) -> int:
        return 0

    @glfns.imported
    def glActiveTexture(self, texture: int) -> int:
        return 0

    @glfns.imported
    def glCheckFramebufferStatus(self, target: int) -> int:
        return 0

    @glfns.imported
    def glGenerateMipmap(self, target: int) -> int:
        return 0

    def __init__(self, window: GLWindow, camera: Camera):
        self.window = window
        self.camera = camera
        self.light_dir = Vector3(0.4, 0.6, 1.0)
        self.ambient = 0.25
        self.shininess = 32.0
        self.spec_strength = 0.5
        self.vertex_src = DEFAULT_VERTEX_SRC
        self.fragment_src = DEFAULT_FRAGMENT_SRC
        self.shadows_enabled = 1
        self.shadow_extent = 15.0
        self.shadow_map_size = 2048
        self.point_lights = []
        self.max_lights = 4
        self.program = 0
        self._u_model = -1
        self._u_view_proj = -1
        self._u_light_dir = -1
        self._u_ambient = -1
        self._u_view_pos = -1
        self._u_shininess = -1
        self._u_spec_strength = -1
        self._u_light_view_proj = -1
        self._u_shadow_map = -1
        self._u_shadow_texel_size = -1
        self._u_point_light_pos = -1
        self._u_point_light_color = -1
        self._u_point_light_range = -1
        self._u_point_light_spot_dir = -1
        self._u_point_light_spot_cutoff = -1
        self._u_point_light_count = -1
        self._u_diffuse_tex = -1
        self._u_use_texture = -1
        self._initialized = 0
        self.shadow_program = 0
        self._su_model = -1
        self._su_light_view_proj = -1
        self._shadow_fbo = 0
        self._shadow_tex = 0

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = 1
        self.glEnable(gl.DEPTH_TEST)
        self.glDepthFunc(gl.LESS)
        self.glEnable(gl.CULL_FACE)
        self.glCullFace(gl.BACK)
        self.glFrontFace(gl.CCW)
        self._compile_program()
        if self.shadows_enabled:
            self._compile_shadow_program()
            self._init_shadow_map()

    def recompile_shader(self):
        """Call after changing vertex_src/fragment_src to a custom GLSL
        pair (e.g. from your own _glsl_dsl-authored shader functions)."""
        self._compile_program()

    def _compile_program(self):
        shader_source_ptr: int = gl_resolve(glfns, "glShaderSource")

        vs_id: int = int(self.glCreateShader(gl.VERTEX_SHADER))
        gl.shader_source(shader_source_ptr, vs_id, self.vertex_src)
        self.glCompileShader(vs_id)
        vs_status: list = [0]
        self.glGetShaderiv(vs_id, gl.COMPILE_STATUS, vs_status)
        if vs_status[0] == 0:
            print("GLRenderer3D: vertex shader compile failed")

        fs_id: int = int(self.glCreateShader(gl.FRAGMENT_SHADER))
        gl.shader_source(shader_source_ptr, fs_id, self.fragment_src)
        self.glCompileShader(fs_id)
        fs_status: list = [0]
        self.glGetShaderiv(fs_id, gl.COMPILE_STATUS, fs_status)
        if fs_status[0] == 0:
            print("GLRenderer3D: fragment shader compile failed")

        program: int = int(self.glCreateProgram())
        self.glAttachShader(program, vs_id)
        self.glAttachShader(program, fs_id)
        self.glLinkProgram(program)
        link_status: list = [0]
        self.glGetProgramiv(program, gl.LINK_STATUS, link_status)
        if link_status[0] == 0:
            print("GLRenderer3D: program link failed")
        self.glDeleteShader(vs_id)
        self.glDeleteShader(fs_id)

        self.program = program
        self._u_model = int(self.glGetUniformLocation(program, "model"))
        self._u_view_proj = int(self.glGetUniformLocation(program, "viewProj"))
        self._u_light_dir = int(self.glGetUniformLocation(program, "lightDir"))
        self._u_ambient = int(self.glGetUniformLocation(program, "ambient"))
        self._u_view_pos = int(self.glGetUniformLocation(program, "viewPos"))
        self._u_shininess = int(self.glGetUniformLocation(program, "shininess"))
        self._u_spec_strength = int(self.glGetUniformLocation(program, "specStrength"))
        self._u_light_view_proj = int(self.glGetUniformLocation(program, "lightViewProj"))
        self._u_shadow_map = int(self.glGetUniformLocation(program, "shadowMap"))
        self._u_shadow_texel_size = int(self.glGetUniformLocation(program, "shadowTexelSize"))
        # Array uniforms: query element [0]'s location -- glUniform3fv/
        # glUniform1fv upload starting there, with the driver advancing
        # through the array's contiguous locations for the rest.
        self._u_point_light_pos = int(self.glGetUniformLocation(program, "pointLightPos[0]"))
        self._u_point_light_color = int(self.glGetUniformLocation(program, "pointLightColor[0]"))
        self._u_point_light_range = int(self.glGetUniformLocation(program, "pointLightRange[0]"))
        self._u_point_light_spot_dir = int(self.glGetUniformLocation(program, "pointLightSpotDir[0]"))
        self._u_point_light_spot_cutoff = int(self.glGetUniformLocation(program, "pointLightSpotCutoff[0]"))
        self._u_point_light_count = int(self.glGetUniformLocation(program, "pointLightCount"))
        self._u_diffuse_tex = int(self.glGetUniformLocation(program, "diffuseTex"))
        self._u_use_texture = int(self.glGetUniformLocation(program, "useTexture"))

    def _compile_shadow_program(self):
        """Compiles the depth-only shadow-pass program (SHADOW_VERTEX_SRC/
        SHADOW_FRAGMENT_SRC) -- separate from the main program since it
        has a different vertex shader (no lighting, just light-space
        position) and only needs `model`/`lightViewProj` uniforms."""
        shader_source_ptr: int = gl_resolve(glfns, "glShaderSource")

        vs_id: int = int(self.glCreateShader(gl.VERTEX_SHADER))
        gl.shader_source(shader_source_ptr, vs_id, SHADOW_VERTEX_SRC)
        self.glCompileShader(vs_id)
        vs_status: list = [0]
        self.glGetShaderiv(vs_id, gl.COMPILE_STATUS, vs_status)
        if vs_status[0] == 0:
            print("GLRenderer3D: shadow vertex shader compile failed")

        fs_id: int = int(self.glCreateShader(gl.FRAGMENT_SHADER))
        gl.shader_source(shader_source_ptr, fs_id, SHADOW_FRAGMENT_SRC)
        self.glCompileShader(fs_id)
        fs_status: list = [0]
        self.glGetShaderiv(fs_id, gl.COMPILE_STATUS, fs_status)
        if fs_status[0] == 0:
            print("GLRenderer3D: shadow fragment shader compile failed")

        program: int = int(self.glCreateProgram())
        self.glAttachShader(program, vs_id)
        self.glAttachShader(program, fs_id)
        self.glLinkProgram(program)
        link_status: list = [0]
        self.glGetProgramiv(program, gl.LINK_STATUS, link_status)
        if link_status[0] == 0:
            print("GLRenderer3D: shadow program link failed")
        self.glDeleteShader(vs_id)
        self.glDeleteShader(fs_id)

        self.shadow_program = program
        self._su_model = int(self.glGetUniformLocation(program, "model"))
        self._su_light_view_proj = int(self.glGetUniformLocation(program, "lightViewProj"))

    def _init_shadow_map(self):
        """Allocates the shadow map's depth texture and a framebuffer that
        renders into it (no color attachment -- GL_NONE draw/read buffers,
        since the shadow pass only needs depth)."""
        tex: list = [0]
        self.glGenTextures(1, tex)
        self.glBindTexture(gl.TEXTURE_2D, tex[0])
        self.glTexImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, self.shadow_map_size, self.shadow_map_size, 0, gl.DEPTH_COMPONENT, gl.FLOAT, 0)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_BORDER)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_BORDER)
        border: list[float] = [1.0, 1.0, 1.0, 1.0]
        self.glTexParameterfv(gl.TEXTURE_2D, gl.TEXTURE_BORDER_COLOR, border)

        fbo: list = [0]
        self.glGenFramebuffers(1, fbo)
        self.glBindFramebuffer(gl.FRAMEBUFFER, fbo[0])
        self.glFramebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, tex[0], 0)
        self.glDrawBuffer(gl.NONE)
        self.glReadBuffer(gl.NONE)
        status: int = int(self.glCheckFramebufferStatus(gl.FRAMEBUFFER))
        if status != gl.FRAMEBUFFER_COMPLETE:
            print("GLRenderer3D: shadow framebuffer incomplete")
        self.glBindFramebuffer(gl.FRAMEBUFFER, 0)

        self._shadow_fbo = fbo[0]
        self._shadow_tex = tex[0]

    def _light_view_proj(self) -> Matrix4:
        """Light-space view-projection: an orthographic box of half-extent
        shadow_extent, centered on the scene origin, looking along
        light_dir -- see GLRenderer3D's CHANGELOG entry on why a fixed
        scene-centered box (vs. a camera-following one) was chosen for
        this engine's current single-bounded-scene use case."""
        light_dir_n: Vector3 = self.light_dir.normalized()
        eye_dist: float = self.shadow_extent * 2.0
        eye: Vector3 = light_dir_n * eye_dist
        origin: Vector3 = Vector3(0.0, 0.0, 0.0)
        up: Vector3 = Vector3(0.0, 1.0, 0.0)
        light_dir_y_abs: float = abs(light_dir_n.y)
        if light_dir_y_abs > 0.99:
            up = Vector3(0.0, 0.0, 1.0)
        light_view: Matrix4 = Matrix4.look_at(eye, origin, up)
        e: float = self.shadow_extent
        far_plane: float = self.shadow_extent * 4.0
        light_proj: Matrix4 = Matrix4.ortho(-e, e, -e, e, 0.1, far_plane)
        return light_proj.multiply(light_view)

    def _render_shadow_pass(self, objects: list[SceneObject], light_view_proj: Matrix4):
        self.glBindFramebuffer(gl.FRAMEBUFFER, self._shadow_fbo)
        self.glViewport(0, 0, self.shadow_map_size, self.shadow_map_size)
        self.glClear(gl.DEPTH_BUFFER_BIT)
        self.glUseProgram(self.shadow_program)
        self.glCullFace(gl.FRONT)  # peter-panning fix: cull front faces so the *back* faces cast the depth

        oi: int = 0
        while oi < len(objects):
            obj: SceneObject = objects[oi]
            if obj.mesh.gl_uploaded == 0:
                self._upload_mesh(obj.mesh, obj.colors)
            self._upload_matrix(self._su_model, obj.model)
            self._upload_matrix(self._su_light_view_proj, light_view_proj)
            self.glBindVertexArray(obj.mesh.gl_vao)
            self.glDrawArrays(gl.TRIANGLES, 0, obj.mesh.gl_vertex_count)
            oi = oi + 1

        self.glCullFace(gl.BACK)
        self.glBindFramebuffer(gl.FRAMEBUFFER, 0)

    def _upload_mesh(self, mesh: Mesh, colors: list[int]):
        """Builds the interleaved (pos, normal, color) vertex buffer for
        `mesh` and uploads it to a fresh VBO/VAO, cached on the Mesh
        itself so later frames skip straight to glDrawArrays. `colors` is
        baked into the buffer at upload time (each vertex of triangle i
        gets colors[i]'s RGB) -- changing a SceneObject's colors after its
        mesh has already been uploaded won't be reflected until the cache
        is invalidated (mesh.gl_uploaded = 0) and re-uploaded.
        """
        verts: list = []
        ti: int = 0
        while ti < len(mesh.triangles):
            tri = mesh.triangles[ti]
            i0: int = tri[0]
            i1: int = tri[1]
            i2: int = tri[2]
            base_color: int = colors[ti]
            cr: float = float((base_color >> 16) & 0xFF) / 255.0
            cg: float = float((base_color >> 8) & 0xFF) / 255.0
            cb: float = float(base_color & 0xFF) / 255.0

            idxs: list[int] = [i0, i1, i2]
            us: list[float] = [mesh.tri_u0[ti], mesh.tri_u1[ti], mesh.tri_u2[ti]]
            vs: list[float] = [mesh.tri_v0[ti], mesh.tri_v1[ti], mesh.tri_v2[ti]]
            vi: int = 0
            while vi < 3:
                vidx: int = idxs[vi]
                p: Vector3 = mesh.vertices[vidx]
                n: Vector3 = mesh.vertex_normals[vidx]
                verts.append(p.x)
                verts.append(p.y)
                verts.append(p.z)
                verts.append(n.x)
                verts.append(n.y)
                verts.append(n.z)
                verts.append(cr)
                verts.append(cg)
                verts.append(cb)
                verts.append(us[vi])
                verts.append(vs[vi])
                vi = vi + 1
            ti = ti + 1

        vao: list = [0]
        self.glGenVertexArrays(1, vao)
        self.glBindVertexArray(vao[0])

        vbo: list = [0]
        self.glGenBuffers(1, vbo)
        self.glBindBuffer(gl.ARRAY_BUFFER, vbo[0])
        self.glBufferData(gl.ARRAY_BUFFER, len(verts) * 8, verts, gl.STATIC_DRAW)

        stride: int = 11 * 8  # 11 floats/vertex, 8 bytes/double (GL_DOUBLE)
        self.glVertexAttribPointer(0, 3, gl.DOUBLE, 0, stride, 0)
        self.glEnableVertexAttribArray(0)
        self.glVertexAttribPointer(1, 3, gl.DOUBLE, 0, stride, 24)
        self.glEnableVertexAttribArray(1)
        self.glVertexAttribPointer(2, 3, gl.DOUBLE, 0, stride, 48)
        self.glEnableVertexAttribArray(2)
        self.glVertexAttribPointer(3, 2, gl.DOUBLE, 0, stride, 72)
        self.glEnableVertexAttribArray(3)

        mesh.gl_vao = vao[0]
        mesh.gl_vbo = vbo[0]
        mesh.gl_vertex_count = len(mesh.triangles) * 3
        mesh.gl_uploaded = 1

    def _upload_texture(self, texture: Texture):
        """Uploads texture.pixels (Texture's flat 0x00RRGGBB list[int]) to
        a fresh GL_TEXTURE_2D, cached on the Texture itself the same way
        _upload_mesh caches a VAO/VBO on its Mesh. Packs through
        _pack_rgba8 first -- glTexImage2D(..., GL_RGBA, GL_UNSIGNED_BYTE,
        ...) reads 4 tightly-packed bytes/pixel, not one 64-bit asmpython
        int/pixel (see _pack_rgba8's own comment). The actual upload call
        goes through _gl_tex_image_2d_data (a hand-marshalled call via its
        resolved pointer) rather than the @glfns.imported glTexImage2D
        stub above, since that stub's `data: int` only fits the NULL-data
        case (see _init_shadow_map) -- see _gl_tex_image_2d_data's own
        comment for why one stub can't serve both shapes."""
        count: int = texture.width * texture.height
        slots: int = count // 2 + count % 2 + 1
        packed: list = self._zero_slots(slots)
        _pack_rgba8(texture.pixels, packed, count)

        tex: list = [0]
        self.glGenTextures(1, tex)
        self.glBindTexture(gl.TEXTURE_2D, tex[0])
        tex_image_2d_ptr: int = gl_resolve(glfns, "glTexImage2D")
        _gl_tex_image_2d_data(tex_image_2d_ptr, gl.TEXTURE_2D, 0, gl.RGBA, texture.width, texture.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, packed)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
        self.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)
        self.glGenerateMipmap(gl.TEXTURE_2D)

        texture.gl_tex = tex[0]
        texture.gl_uploaded = 1

    def _upload_matrix(self, location: int, m: Matrix4):
        """glUniformMatrix4fv needs 16 packed 32-bit floats, but Matrix4.m
        is a list[float] (16 doubles) -- pack via _pack_floats32 first
        (see its own comment for why this can't go directly through
        glUniformMatrix4fv)."""
        packed: list = [0, 0, 0, 0, 0, 0, 0, 0]
        _pack_floats32(m.m, packed, 16)
        self.glUniformMatrix4fv(location, 1, gl.TRUE, packed)

    def _upload_point_lights(self):
        """Uploads renderer.point_lights (capped at max_lights) into the
        shader's pointLight* array uniforms via glUniform3fv/glUniform1fv,
        packing through _pack_floats32 the same way _upload_matrix does --
        each upload's source list[float] is always asmpython doubles, and
        every *fv GL entry point reads packed 32-bit floats regardless of
        whether it's a matrix, vec3, or plain float array."""
        n: int = len(self.point_lights)
        if n > self.max_lights:
            n = self.max_lights
        self.glUniform1i(self._u_point_light_count, n)
        if n == 0:
            return

        pos_doubles: list[float] = []
        color_doubles: list[float] = []
        range_doubles: list[float] = []
        spot_dir_doubles: list[float] = []
        spot_cutoff_doubles: list[float] = []
        i: int = 0
        while i < n:
            light: PointLight = self.point_lights[i]
            pos_doubles.append(light.position.x)
            pos_doubles.append(light.position.y)
            pos_doubles.append(light.position.z)
            color_doubles.append(light.color.x)
            color_doubles.append(light.color.y)
            color_doubles.append(light.color.z)
            range_doubles.append(light.range)
            spot_dir_doubles.append(light.spot_direction.x)
            spot_dir_doubles.append(light.spot_direction.y)
            spot_dir_doubles.append(light.spot_direction.z)
            spot_cutoff_doubles.append(light.spot_cutoff)
            i = i + 1

        vec3_slots: int = (n * 3) // 2 + (n * 3) % 2 + 1
        pos_packed: list = self._zero_slots(vec3_slots)
        _pack_floats32(pos_doubles, pos_packed, n * 3)
        self.glUniform3fv(self._u_point_light_pos, n, pos_packed)

        color_packed: list = self._zero_slots(vec3_slots)
        _pack_floats32(color_doubles, color_packed, n * 3)
        self.glUniform3fv(self._u_point_light_color, n, color_packed)

        spot_dir_packed: list = self._zero_slots(vec3_slots)
        _pack_floats32(spot_dir_doubles, spot_dir_packed, n * 3)
        self.glUniform3fv(self._u_point_light_spot_dir, n, spot_dir_packed)

        scalar_slots: int = n // 2 + n % 2 + 1
        range_packed: list = self._zero_slots(scalar_slots)
        _pack_floats32(range_doubles, range_packed, n)
        self.glUniform1fv(self._u_point_light_range, n, range_packed)

        spot_cutoff_packed: list = self._zero_slots(scalar_slots)
        _pack_floats32(spot_cutoff_doubles, spot_cutoff_packed, n)
        self.glUniform1fv(self._u_point_light_spot_cutoff, n, spot_cutoff_packed)

    def _zero_slots(self, count: int) -> list:
        result: list = []
        i: int = 0
        while i < count:
            result.append(0)
            i = i + 1
        return result

    def _draw_object(self, mesh: Mesh, model: Matrix4, colors: list[int], texture: Texture, view_proj: Matrix4, light_view_proj: Matrix4):
        if mesh.gl_uploaded == 0:
            self._upload_mesh(mesh, colors)

        self._upload_matrix(self._u_model, model)
        self._upload_matrix(self._u_view_proj, view_proj)
        self._upload_matrix(self._u_light_view_proj, light_view_proj)

        if texture is None:
            self.glUniform1f(self._u_use_texture, 0.0)
        else:
            if texture.gl_uploaded == 0:
                self._upload_texture(texture)
            self.glActiveTexture(gl.TEXTURE1)
            self.glBindTexture(gl.TEXTURE_2D, texture.gl_tex)
            self.glUniform1i(self._u_diffuse_tex, 1)
            self.glUniform1f(self._u_use_texture, 1.0)

        self.glBindVertexArray(mesh.gl_vao)
        self.glDrawArrays(gl.TRIANGLES, 0, mesh.gl_vertex_count)

    def render_scene(self, objects: list[SceneObject]):
        self._ensure_initialized()

        light_view_proj: Matrix4 = Matrix4.identity()
        if self.shadows_enabled:
            light_view_proj = self._light_view_proj()
            self._render_shadow_pass(objects, light_view_proj)

        self.glViewport(0, 0, self.window.width, self.window.height)
        self.glClearColor(0.05, 0.05, 0.08, 1.0)
        self.glClear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        self.glUseProgram(self.program)
        light_dir_n: Vector3 = self.light_dir.normalized()
        self.glUniform3f(self._u_light_dir, light_dir_n.x, light_dir_n.y, light_dir_n.z)
        self.glUniform1f(self._u_ambient, self.ambient)
        self.glUniform3f(self._u_view_pos, self.camera.position.x, self.camera.position.y, self.camera.position.z)
        self.glUniform1f(self._u_shininess, self.shininess)
        self.glUniform1f(self._u_spec_strength, self.spec_strength)

        if self.shadows_enabled:
            texel_size: float = 1.0 / float(self.shadow_map_size)
            self.glUniform1f(self._u_shadow_texel_size, texel_size)
            self.glActiveTexture(gl.TEXTURE0)
            self.glBindTexture(gl.TEXTURE_2D, self._shadow_tex)
            self.glUniform1i(self._u_shadow_map, 0)
        else:
            self.glUniform1f(self._u_shadow_texel_size, 0.0)

        self._upload_point_lights()

        proj: Matrix4 = self.camera.projection_matrix()
        view: Matrix4 = self.camera.view_matrix()
        view_proj: Matrix4 = proj.multiply(view)

        oi: int = 0
        while oi < len(objects):
            obj: SceneObject = objects[oi]
            self._draw_object(obj.mesh, obj.model, obj.colors, obj.texture, view_proj, light_view_proj)
            oi = oi + 1

    def render_solid(self, mesh: Mesh, model: Matrix4, colors: list[int]):
        """Single-object convenience wrapper around render_scene(), for
        the common one-mesh-per-frame case (matches Renderer3D.render_solid's
        call shape)."""
        obj = SceneObject(mesh, model, colors, None)
        self.render_scene([obj])

    def update(self) -> int:
        """Present the current frame and pump events -- call once per
        frame after render_scene(), matching Renderer3D's call shape
        (renderer.window.update() there; this is the same operation, just
        also reachable directly off the renderer since GLWindow.update()
        does the identical swap_window()+poll_event() work)."""
        return self.window.update()
