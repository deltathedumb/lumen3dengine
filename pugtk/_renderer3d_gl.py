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
from ._gl_shaders_compiled import DEFAULT_VERTEX_SRC, DEFAULT_FRAGMENT_SRC

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
    vertex_src: str
    fragment_src: str

    program: int
    _u_model: int
    _u_view_proj: int
    _u_light_dir: int
    _u_ambient: int
    _initialized: int

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

    def __init__(self, window: GLWindow, camera: Camera):
        self.window = window
        self.camera = camera
        self.light_dir = Vector3(0.4, 0.6, 1.0)
        self.ambient = 0.25
        self.vertex_src = DEFAULT_VERTEX_SRC
        self.fragment_src = DEFAULT_FRAGMENT_SRC
        self.program = 0
        self._u_model = -1
        self._u_view_proj = -1
        self._u_light_dir = -1
        self._u_ambient = -1
        self._initialized = 0

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
                vi = vi + 1
            ti = ti + 1

        vao: list = [0]
        self.glGenVertexArrays(1, vao)
        self.glBindVertexArray(vao[0])

        vbo: list = [0]
        self.glGenBuffers(1, vbo)
        self.glBindBuffer(gl.ARRAY_BUFFER, vbo[0])
        self.glBufferData(gl.ARRAY_BUFFER, len(verts) * 8, verts, gl.STATIC_DRAW)

        stride: int = 9 * 8  # 9 floats/vertex, 8 bytes/double (GL_DOUBLE)
        self.glVertexAttribPointer(0, 3, gl.DOUBLE, 0, stride, 0)
        self.glEnableVertexAttribArray(0)
        self.glVertexAttribPointer(1, 3, gl.DOUBLE, 0, stride, 24)
        self.glEnableVertexAttribArray(1)
        self.glVertexAttribPointer(2, 3, gl.DOUBLE, 0, stride, 48)
        self.glEnableVertexAttribArray(2)

        mesh.gl_vao = vao[0]
        mesh.gl_vbo = vbo[0]
        mesh.gl_vertex_count = len(mesh.triangles) * 3
        mesh.gl_uploaded = 1

    def _upload_matrix(self, location: int, m: Matrix4):
        """glUniformMatrix4fv needs 16 packed 32-bit floats, but Matrix4.m
        is a list[float] (16 doubles) -- pack via _pack_floats32 first
        (see its own comment for why this can't go directly through
        glUniformMatrix4fv)."""
        packed: list = [0, 0, 0, 0, 0, 0, 0, 0]
        _pack_floats32(m.m, packed, 16)
        self.glUniformMatrix4fv(location, 1, gl.TRUE, packed)

    def _draw_object(self, mesh: Mesh, model: Matrix4, colors: list[int], view_proj: Matrix4):
        if mesh.gl_uploaded == 0:
            self._upload_mesh(mesh, colors)

        self._upload_matrix(self._u_model, model)
        self._upload_matrix(self._u_view_proj, view_proj)

        self.glBindVertexArray(mesh.gl_vao)
        self.glDrawArrays(gl.TRIANGLES, 0, mesh.gl_vertex_count)

    def render_scene(self, objects: list[SceneObject]):
        self._ensure_initialized()

        self.glViewport(0, 0, self.window.width, self.window.height)
        self.glClearColor(0.05, 0.05, 0.08, 1.0)
        self.glClear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        self.glUseProgram(self.program)
        self.glUniform3f(self._u_light_dir, self.light_dir.x, self.light_dir.y, self.light_dir.z)
        self.glUniform1f(self._u_ambient, self.ambient)

        proj: Matrix4 = self.camera.projection_matrix()
        view: Matrix4 = self.camera.view_matrix()
        view_proj: Matrix4 = proj.multiply(view)

        oi: int = 0
        while oi < len(objects):
            obj: SceneObject = objects[oi]
            self._draw_object(obj.mesh, obj.model, obj.colors, view_proj)
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
