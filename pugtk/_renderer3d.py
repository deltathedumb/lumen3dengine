from typing import Callable

import lumen

from ._vector import Vector2, Vector3
from ._shapes import PointShape
from ._matrix import Matrix4
from ._mesh import Mesh
from ._camera import Camera
from ._window import Window
from ._shading import ShaderContext, unlit_shader
from ._texture import Texture
from ._scene import SceneObject


class _ClipVert:
    """One clip-space vertex plus every attribute the rasterizer needs to
    interpolate, carried together through near-plane clipping so a clipped
    triangle's new edge vertices get correctly lerped attributes instead of
    just a lerped position.

    A plain class (not parallel flat lists like the rest of this module
    deliberately uses elsewhere) because clipping is inherently a small,
    fixed-arity, append/lerp-heavy operation on *one* triangle at a time --
    not a per-pixel hot loop -- so the simpler shape is worth it here.
    """

    def __init__(
        self,
        cx: float, cy: float, cz: float, cw: float,
        wx: float, wy: float, wz: float,
        lx: float, ly: float, lz: float,
        nx: float, ny: float, nz: float,
        u: float, v: float,
    ):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.cw = cw
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.u = u
        self.v = v


def _lerp_clip_vert(a: _ClipVert, b: _ClipVert, t: float) -> _ClipVert:
    return _ClipVert(
        a.cx + (b.cx - a.cx) * t,
        a.cy + (b.cy - a.cy) * t,
        a.cz + (b.cz - a.cz) * t,
        a.cw + (b.cw - a.cw) * t,
        a.wx + (b.wx - a.wx) * t,
        a.wy + (b.wy - a.wy) * t,
        a.wz + (b.wz - a.wz) * t,
        a.lx + (b.lx - a.lx) * t,
        a.ly + (b.ly - a.ly) * t,
        a.lz + (b.lz - a.lz) * t,
        a.nx + (b.nx - a.nx) * t,
        a.ny + (b.ny - a.ny) * t,
        a.nz + (b.nz - a.nz) * t,
        a.u + (b.u - a.u) * t,
        a.v + (b.v - a.v) * t,
    )


def _near_plane_dist(v: _ClipVert) -> float:
    # Signed distance from the near clip plane (z + w == 0, this
    # projection's convention -- see Matrix4.perspective()). Positive is
    # inside (visible side); a vertex exactly on the camera (w == 0) is
    # treated as just outside to avoid a divide-by-zero in the perspective
    # divide downstream.
    return v.cz + v.cw


def _clip_near(verts: list[_ClipVert]) -> list[_ClipVert]:
    """Sutherland-Hodgman clip of one triangle (3 verts, in order) against
    the near plane only. Far/left/right/top/bottom are left to screen-space
    clamping in the rasterizer (cheap, and this engine's frustums are small
    enough that the only plane that actually produces visible artifacts --
    a vertex behind the eye wrapping the whole triangle around the screen --
    is the near plane).

    Returns 0 verts (fully behind), 3 (unclipped or clipped to a triangle),
    or 4 (clipped to a quad -- the caller fans this into 2 triangles).
    """
    out: list[_ClipVert] = []
    n: int = len(verts)
    i: int = 0
    while i < n:
        cur: _ClipVert = verts[i]
        prev: _ClipVert = verts[(i - 1 + n) % n]
        cur_d: float = _near_plane_dist(cur)
        prev_d: float = _near_plane_dist(prev)
        cur_in: bool = cur_d >= 0.0
        prev_in: bool = prev_d >= 0.0

        if cur_in:
            if not prev_in:
                t: float = prev_d / (prev_d - cur_d)
                out.append(_lerp_clip_vert(prev, cur, t))
            out.append(cur)
        else:
            if prev_in:
                t: float = prev_d / (prev_d - cur_d)
                out.append(_lerp_clip_vert(prev, cur, t))
        i = i + 1
    return out


class Renderer3D:
    window: Window
    camera: Camera
    shader: Callable[[ShaderContext], int]
    pixel_shader: Callable[[ShaderContext], int]
    light_dir: Vector3
    light_color: int
    ambient: float
    specular: float
    shininess: float
    # Material params for pbr_shader()/pbr_rim_shader() (see _shading.py) --
    # ignored by every other shader, same as specular/shininess being
    # phong_shader-only. Per-renderer rather than per-face for now, same
    # limitation specular/shininess already have.
    metallic: float
    roughness: float
    no_texture: Texture
    # Supersample factor: triangles rasterize into an internal buffer
    # `supersample` times wider/taller than the window, which is then
    # box-filter downsampled into window.pixbuf once per render_* call.
    # This is the standard SSAA antialiasing tradeoff -- N times the pixels
    # rasterized and shaded for an NxN sample grid per output pixel -- in
    # exchange for smooth triangle edges instead of the jagged/speckled
    # look a 1-sample-per-pixel rasterizer produces (most visible as stray
    # single-pixel speckling along a near-grazing-angle edge, since whether
    # a borderline pixel's *single* sample point happens to fall inside or
    # outside the triangle flips almost at random pixel-to-pixel there).
    # 1 disables antialiasing entirely (every sample == the output pixel).
    supersample: int
    # Per-sample depth buffer (camera-space 1/w, flat row-major, sized
    # supersample*width x supersample*height). Cleared to 0.0 -- the
    # smallest possible 1/w for anything in front of the near plane -- at
    # the start of every render_* call, so each frame's first write to a
    # sample always passes the depth test.
    _depth: list[float]
    _depth_w: int
    _depth_h: int
    # Per-sample color buffer, same dimensions as _depth. A real PixelBuffer
    # (not a plain list[int]) since downsampling reads it back per output
    # pixel via .get(), matching the packing/clamping PixelBuffer already does.
    _samples: lumen.PixelBuffer
    _samples_w: int
    _samples_h: int

    def __init__(self, window: Window, camera: Camera):
        self.window = window
        self.camera = camera
        # Plain top-level functions only (no closures): asmpython's "this
        # value is a closure" type tag is derived by tracing an assignment's
        # RHS back to a literal closure-factory call, not a real type that
        # flows through parameters/attributes -- storing a closure (a
        # nested function that captured a free variable) here crashes when
        # called. Plain functions with no captured state don't hit that
        # path, so per-shader config (light direction, ambient, ...) is
        # carried on ShaderContext / these attributes instead of being
        # closed over.
        self.shader = unlit_shader
        self.pixel_shader = unlit_shader
        self.light_dir = Vector3(0.0, 0.0, 1.0)
        self.light_color = 0x00FFFFFF
        self.ambient = 0.3
        self.specular = 0.5
        self.shininess = 32.0
        self.metallic = 0.0
        self.roughness = 0.5
        # render_solid() has no per-pixel UV to sample (it's one shader call
        # per face), so its ShaderContext just gets this placeholder plus a
        # centroid UV of 0.5, 0.5 -- textured_shader still "works" there,
        # just samples a single texel for the whole face.
        self.no_texture = Texture.solid(1, 1, 0)
        # Off by default: SSAA rasterizes and shades supersample**2 times
        # the pixels, which is a real (not free) cost on a software
        # rasterizer with no GPU behind it -- opt in with
        # `renderer.supersample = 2` (or higher) once a scene's perf
        # budget can afford smoother edges.
        self.supersample = 1
        self._depth = []
        self._depth_w = 0
        self._depth_h = 0
        self._samples = lumen.PixelBuffer(1, 1)
        self._samples_w = 0
        self._samples_h = 0

    def _ensure_buffers(self):
        ss: int = self.supersample
        if ss < 1:
            ss = 1
        if ss == 1:
            # No antialiasing: rasterize directly into window.pixbuf
            # instead of a separate same-size sample buffer, so there's no
            # per-frame full-buffer resolve/copy pass at all -- this path
            # costs exactly what the old non-supersampled renderer did.
            self._samples = self.window.pixbuf
            w: int = self.window.width
            h: int = self.window.height
        else:
            w: int = self.window.width * ss
            h: int = self.window.height * ss
            if w != self._samples_w or h != self._samples_h:
                self._samples = lumen.PixelBuffer(w, h)
            self._samples.clear(0)
        self._samples_w = w
        self._samples_h = h

        if w == self._depth_w and h == self._depth_h and len(self._depth) == w * h:
            i: int = 0
            n: int = len(self._depth)
            while i < n:
                self._depth[i] = 0.0
                i = i + 1
            return
        self._depth = []
        total: int = w * h
        i: int = 0
        while i < total:
            self._depth.append(0.0)
            i = i + 1
        self._depth_w = w
        self._depth_h = h

    def _resolve(self):
        """Box-filter downsample _samples (supersample*W x supersample*H)
        into window.pixbuf (W x H): each output pixel is the average of its
        supersample*supersample backing sample block.

        No-op when supersample == 1: _ensure_buffers() made _samples the
        exact same PixelBuffer object as window.pixbuf in that case, so the
        rasterizer already wrote the final pixels directly -- there's
        nothing left to copy.
        """
        ss: int = self.supersample
        if ss < 1:
            ss = 1
        if ss == 1:
            return

        sample_count: int = ss * ss
        out_y: int = 0
        while out_y < self.window.height:
            base_y: int = out_y * ss
            out_x: int = 0
            while out_x < self.window.width:
                base_x: int = out_x * ss
                sum_r: int = 0
                sum_g: int = 0
                sum_b: int = 0
                dy: int = 0
                while dy < ss:
                    dx: int = 0
                    while dx < ss:
                        c: int = self._samples.get(base_x + dx, base_y + dy)
                        sum_r = sum_r + ((c >> 16) & 0xFF)
                        sum_g = sum_g + ((c >> 8) & 0xFF)
                        sum_b = sum_b + (c & 0xFF)
                        dx = dx + 1
                    dy = dy + 1
                avg_r: int = sum_r // sample_count
                avg_g: int = sum_g // sample_count
                avg_b: int = sum_b // sample_count
                self.window.pixbuf.set(out_x, out_y, (avg_r << 16) | (avg_g << 8) | avg_b)
                out_x = out_x + 1
            out_y = out_y + 1

    def render_wireframe(self, mesh: Mesh, model: Matrix4, color: int):
        view: Matrix4 = self.camera.view_matrix()
        proj: Matrix4 = self.camera.projection_matrix()
        vp: Matrix4 = proj.multiply(view)
        mvp: Matrix4 = vp.multiply(model)

        screen: list[Vector2] = []
        visible: list[int] = []
        half_w: float = self.window.width / 2.0
        half_h: float = self.window.height / 2.0

        for v in mesh.vertices:
            clip = mvp.transform_point(v)
            cx: float = clip[0]
            cy: float = clip[1]
            cw: float = clip[3]
            if cw <= 0.0:
                visible.append(0)
                screen.append(Vector2(0, 0))
                continue
            ndc_x: float = cx / cw
            ndc_y: float = cy / cw
            sx: int = int(half_w + ndc_x * half_w)
            sy: int = int(half_h - ndc_y * half_h)
            screen.append(Vector2(sx, sy))
            visible.append(1)

        for edge in mesh.edges:
            i0: int = edge[0]
            i1: int = edge[1]
            if visible[i0] == 0 or visible[i1] == 0:
                continue
            edge_line = PointShape([screen[i0], screen[i1]])
            self.window.draw(edge_line, color)

    def render_solid(self, mesh: Mesh, model: Matrix4, colors: list[int]):
        """colors[i] is the base color for mesh.triangles[i]; self.shader
        (set to unlit_shader, lambert_shader, phong_shader, or any plain
        function matching Callable[[ShaderContext], int]) decides the final
        flat color, one shader call per triangle.

        Goes through the same clip -> rasterize -> z-test pipeline as
        render_solid_per_pixel(), just with pixel_shader fixed to a
        constant-color stamp instead of calling self.pixel_shader -- so
        flat-shaded triangles still depth-test correctly against (and get
        correctly occluded by) per-pixel-shaded ones in the same frame.
        """
        self._ensure_buffers()
        self._render_mesh(mesh, model, colors, self.no_texture, self.shader, True)
        self._resolve()

    def render_solid_per_pixel(
        self,
        mesh: Mesh,
        model: Matrix4,
        colors: list[int],
        texture: Texture,
    ):
        """Like render_solid(), but self.pixel_shader is called once per
        pixel inside each visible triangle (with ShaderContext.world_pos,
        .local_pos, .normal, .tex_u/.tex_v, and .screen_x/.screen_y all
        genuinely interpolated across the triangle) instead of once per
        triangle. Needed for effects that vary across a single face:
        procedural patterns, Gouraud-shaded curvature, textures, fog
        evaluated per-pixel rather than from one averaged centroid.
        """
        self._ensure_buffers()
        self._render_mesh(mesh, model, colors, texture, self.pixel_shader, False)
        self._resolve()

    def render_scene(self, objects: list[SceneObject]):
        """Renders every SceneObject's mesh against one shared depth
        buffer, so occlusion between objects (and between a single object's
        own front/back triangles) falls out of the per-pixel z-test instead
        of needing any cross-object draw-order sort -- unlike the old
        painter's-algorithm centroid sort this replaces, two
        interpenetrating meshes (or a triangle that's farther at one corner
        and nearer at another) now resolve correctly per pixel.

        Draw order across objects no longer matters for correctness, so
        objects are simply rendered in list order.
        """
        self._ensure_buffers()
        oi: int = 0
        while oi < len(objects):
            obj: SceneObject = objects[oi]
            self._render_mesh(obj.mesh, obj.model, obj.colors, obj.texture, self.pixel_shader, False)
            oi = oi + 1
        self._resolve()

    def _render_mesh(
        self,
        mesh: Mesh,
        model: Matrix4,
        colors: list[int],
        texture: Texture,
        pixel_shader: Callable[[ShaderContext], int],
        flat_per_triangle: bool,
    ):
        proj: Matrix4 = self.camera.projection_matrix()
        view: Matrix4 = self.camera.view_matrix()
        vp: Matrix4 = proj.multiply(view)
        mvp: Matrix4 = vp.multiply(model)

        ctx = ShaderContext(
            Vector3(0.0, 0.0, 1.0),
            Vector3(0.0, 0.0, 0.0),
            Vector3(0.0, 0.0, 0.0),
            self.camera.position,
            self.light_dir,
            self.light_color,
            self.ambient,
            self.specular,
            self.shininess,
            0,
            texture,
            0.0,
            0.0,
            0,
            0,
            self.metallic,
            self.roughness,
        )

        ti: int = 0
        while ti < len(mesh.triangles):
            tri = mesh.triangles[ti]
            i0: int = tri[0]
            i1: int = tri[1]
            i2: int = tri[2]
            v0: Vector3 = mesh.vertices[i0]
            v1: Vector3 = mesh.vertices[i1]
            v2: Vector3 = mesh.vertices[i2]

            c0 = mvp.transform_point(v0)
            c1 = mvp.transform_point(v1)
            c2 = mvp.transform_point(v2)

            w0 = model.transform_point(v0)
            w1 = model.transform_point(v1)
            w2 = model.transform_point(v2)

            n0: Vector3 = mesh.vertex_normals[i0]
            n1: Vector3 = mesh.vertex_normals[i1]
            n2: Vector3 = mesh.vertex_normals[i2]
            wn0 = model.transform_direction(n0)
            wn1 = model.transform_direction(n1)
            wn2 = model.transform_direction(n2)

            cv0 = _ClipVert(
                c0[0], c0[1], c0[2], c0[3],
                w0[0], w0[1], w0[2],
                v0.x, v0.y, v0.z,
                wn0[0], wn0[1], wn0[2],
                mesh.tri_u0[ti], mesh.tri_v0[ti],
            )
            cv1 = _ClipVert(
                c1[0], c1[1], c1[2], c1[3],
                w1[0], w1[1], w1[2],
                v1.x, v1.y, v1.z,
                wn1[0], wn1[1], wn1[2],
                mesh.tri_u1[ti], mesh.tri_v1[ti],
            )
            cv2 = _ClipVert(
                c2[0], c2[1], c2[2], c2[3],
                w2[0], w2[1], w2[2],
                v2.x, v2.y, v2.z,
                wn2[0], wn2[1], wn2[2],
                mesh.tri_u2[ti], mesh.tri_v2[ti],
            )

            clipped: list[_ClipVert] = _clip_near([cv0, cv1, cv2])
            base_color: int = colors[ti]

            if len(clipped) == 3:
                self._raster_clip_tri(clipped[0], clipped[1], clipped[2], base_color, pixel_shader, flat_per_triangle, ctx)
            elif len(clipped) == 4:
                self._raster_clip_tri(clipped[0], clipped[1], clipped[2], base_color, pixel_shader, flat_per_triangle, ctx)
                self._raster_clip_tri(clipped[0], clipped[2], clipped[3], base_color, pixel_shader, flat_per_triangle, ctx)
            ti = ti + 1

    def _raster_clip_tri(
        self,
        cv0: _ClipVert,
        cv1: _ClipVert,
        cv2: _ClipVert,
        base_color: int,
        pixel_shader: Callable[[ShaderContext], int],
        flat_per_triangle: bool,
        ctx: ShaderContext,
    ):
        half_w: float = self._samples_w / 2.0
        half_h: float = self._samples_h / 2.0

        iw0: float = 1.0 / cv0.cw
        iw1: float = 1.0 / cv1.cw
        iw2: float = 1.0 / cv2.cw

        sx0: float = half_w + (cv0.cx * iw0) * half_w
        sy0: float = half_h - (cv0.cy * iw0) * half_h
        sx1: float = half_w + (cv1.cx * iw1) * half_w
        sy1: float = half_h - (cv1.cy * iw1) * half_h
        sx2: float = half_w + (cv2.cx * iw2) * half_w
        sy2: float = half_h - (cv2.cy * iw2) * half_h

        # Twice the signed screen-space area. <= 0 means the triangle winds
        # clockwise in screen space (i.e. faces away from the camera, given
        # this engine's outward-CCW winding convention) -- backface culled
        # here instead of via a world-space normal/to-camera dot product, so
        # triangles produced by near-plane clipping (which don't have a
        # single original face normal anymore) cull correctly too.
        area2: float = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0)
        if area2 >= 0.0:
            return

        # Hand-rolled float min/max instead of the min()/max() builtins:
        # asmpython's 2+-arg min/max lowers to an integer cmp/cmovl, which
        # compares these floats' raw bit patterns instead of their values
        # (confirmed compiler codegen bug) -- silently corrupting the
        # bounding box into garbage.
        fminx: float = sx0
        if sx1 < fminx:
            fminx = sx1
        if sx2 < fminx:
            fminx = sx2
        fmaxx: float = sx0
        if sx1 > fmaxx:
            fmaxx = sx1
        if sx2 > fmaxx:
            fmaxx = sx2
        fminy: float = sy0
        if sy1 < fminy:
            fminy = sy1
        if sy2 < fminy:
            fminy = sy2
        fmaxy: float = sy0
        if sy1 > fmaxy:
            fmaxy = sy1
        if sy2 > fmaxy:
            fmaxy = sy2

        min_x: int = int(fminx)
        max_x: int = int(fmaxx) + 1
        min_y: int = int(fminy)
        max_y: int = int(fmaxy) + 1
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > self._samples_w:
            max_x = self._samples_w
        if max_y > self._samples_h:
            max_y = self._samples_h
        if min_x >= max_x or min_y >= max_y:
            return

        inv_area2: float = 1.0 / area2

        wow0_x: float = cv0.wx * iw0
        wow0_y: float = cv0.wy * iw0
        wow0_z: float = cv0.wz * iw0
        wow1_x: float = cv1.wx * iw1
        wow1_y: float = cv1.wy * iw1
        wow1_z: float = cv1.wz * iw1
        wow2_x: float = cv2.wx * iw2
        wow2_y: float = cv2.wy * iw2
        wow2_z: float = cv2.wz * iw2

        low0_x: float = cv0.lx * iw0
        low0_y: float = cv0.ly * iw0
        low0_z: float = cv0.lz * iw0
        low1_x: float = cv1.lx * iw1
        low1_y: float = cv1.ly * iw1
        low1_z: float = cv1.lz * iw1
        low2_x: float = cv2.lx * iw2
        low2_y: float = cv2.ly * iw2
        low2_z: float = cv2.lz * iw2

        no0_x: float = cv0.nx * iw0
        no0_y: float = cv0.ny * iw0
        no0_z: float = cv0.nz * iw0
        no1_x: float = cv1.nx * iw1
        no1_y: float = cv1.ny * iw1
        no1_z: float = cv1.nz * iw1
        no2_x: float = cv2.nx * iw2
        no2_y: float = cv2.ny * iw2
        no2_z: float = cv2.nz * iw2

        uo0: float = cv0.u * iw0
        vo0: float = cv0.v * iw0
        uo1: float = cv1.u * iw1
        vo1: float = cv1.v * iw1
        uo2: float = cv2.u * iw2
        vo2: float = cv2.v * iw2

        flat_color: int = base_color
        if flat_per_triangle:
            cx: float = (cv0.wx + cv1.wx + cv2.wx) / 3.0
            cy: float = (cv0.wy + cv1.wy + cv2.wy) / 3.0
            cz: float = (cv0.wz + cv1.wz + cv2.wz) / 3.0
            lcx: float = (cv0.lx + cv1.lx + cv2.lx) / 3.0
            lcy: float = (cv0.ly + cv1.ly + cv2.ly) / 3.0
            lcz: float = (cv0.lz + cv1.lz + cv2.lz) / 3.0
            ncx: float = cv0.nx + cv1.nx + cv2.nx
            ncy: float = cv0.ny + cv1.ny + cv2.ny
            ncz: float = cv0.nz + cv1.nz + cv2.nz

            ctx.world_pos.x = cx
            ctx.world_pos.y = cy
            ctx.world_pos.z = cz
            ctx.local_pos.x = lcx
            ctx.local_pos.y = lcy
            ctx.local_pos.z = lcz
            ctx.normal.x = ncx
            ctx.normal.y = ncy
            ctx.normal.z = ncz
            ctx.normal.normalize_in_place()
            ctx.tex_u = 0.5
            ctx.tex_v = 0.5
            ctx.base_color = base_color
            flat_color = pixel_shader(ctx)

        depth: list[float] = self._depth
        depth_w: int = self._depth_w
        samples: lumen.PixelBuffer = self._samples
        ss: int = self.supersample
        if ss < 1:
            ss = 1
        ctx_world_pos: Vector3 = ctx.world_pos
        ctx_local_pos: Vector3 = ctx.local_pos
        ctx_normal: Vector3 = ctx.normal

        py: int = min_y
        while py < max_y:
            fy: float = float(py) + 0.5
            px: int = min_x
            while px < max_x:
                fx: float = float(px) + 0.5

                w0: float = (sx1 - fx) * (sy2 - fy) - (sx2 - fx) * (sy1 - fy)
                w1: float = (sx2 - fx) * (sy0 - fy) - (sx0 - fx) * (sy2 - fy)
                w2: float = (sx0 - fx) * (sy1 - fy) - (sx1 - fx) * (sy0 - fy)

                if w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0:
                    b0: float = w0 * inv_area2
                    b1: float = w1 * inv_area2
                    b2: float = w2 * inv_area2

                    p_iw: float = b0 * iw0 + b1 * iw1 + b2 * iw2
                    didx: int = py * depth_w + px
                    if p_iw > depth[didx]:
                        depth[didx] = p_iw

                        if flat_per_triangle:
                            samples.set(px, py, flat_color)
                        else:
                            p_w: float = 1.0 / p_iw
                            ctx_world_pos.x = (b0 * wow0_x + b1 * wow1_x + b2 * wow2_x) * p_w
                            ctx_world_pos.y = (b0 * wow0_y + b1 * wow1_y + b2 * wow2_y) * p_w
                            ctx_world_pos.z = (b0 * wow0_z + b1 * wow1_z + b2 * wow2_z) * p_w
                            ctx_local_pos.x = (b0 * low0_x + b1 * low1_x + b2 * low2_x) * p_w
                            ctx_local_pos.y = (b0 * low0_y + b1 * low1_y + b2 * low2_y) * p_w
                            ctx_local_pos.z = (b0 * low0_z + b1 * low1_z + b2 * low2_z) * p_w
                            # Linear interpolation (then dividing by w)
                            # doesn't preserve unit length, so the averaged
                            # normal must be renormalized per pixel --
                            # otherwise lighting intensity
                            # (normal.dot(light_dir)) would be biased by how
                            # far |normal| has drifted from 1.
                            ctx_normal.x = (b0 * no0_x + b1 * no1_x + b2 * no2_x) * p_w
                            ctx_normal.y = (b0 * no0_y + b1 * no1_y + b2 * no2_y) * p_w
                            ctx_normal.z = (b0 * no0_z + b1 * no1_z + b2 * no2_z) * p_w
                            ctx_normal.normalize_in_place()
                            ctx.tex_u = (b0 * uo0 + b1 * uo1 + b2 * uo2) * p_w
                            ctx.tex_v = (b0 * vo0 + b1 * vo1 + b2 * vo2) * p_w
                            # Output-pixel coordinates (sample coords / ss),
                            # not raw supersample-grid coordinates -- a
                            # shader keying off screen_x/screen_y (e.g. a
                            # screen-space dither or scanline effect)
                            # shouldn't see a result that depends on
                            # Renderer3D.supersample.
                            ctx.screen_x = px // ss
                            ctx.screen_y = py // ss
                            ctx.base_color = base_color
                            samples.set(px, py, pixel_shader(ctx))
                px = px + 1
            py = py + 1
