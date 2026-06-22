from typing import Callable

from ._vector import Vector2, Vector3
from ._shapes import PointShape
from ._matrix import Matrix4
from ._mesh import Mesh
from ._camera import Camera
from ._window import Window
from ._shading import ShaderContext, unlit_shader
from ._texture import Texture
from ._scene import SceneObject


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
    no_texture: Texture

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
        # render_solid() has no per-pixel UV to sample (it's one shader call
        # per face), so its ShaderContext just gets this placeholder plus a
        # centroid UV of 0.5, 0.5 -- textured_shader still "works" there,
        # just samples a single texel for the whole face.
        self.no_texture = Texture.solid(1, 1, 0)

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
        """colors[i] is the base color for mesh.faces[i]; self.shader (set
        to unlit_shader, lambert_shader, phong_shader, or any plain
        function matching Callable[[ShaderContext], int]) decides the
        final per-face color.

        Culling, depth-sort, and shading are all done in world space here
        (face normal vs. camera direction for culling, vs. light_dir for
        shading), since the light is defined in world space.
        """
        proj: Matrix4 = self.camera.projection_matrix()
        view: Matrix4 = self.camera.view_matrix()
        vp: Matrix4 = proj.multiply(view)
        mvp: Matrix4 = vp.multiply(model)

        half_w: float = self.window.width / 2.0
        half_h: float = self.window.height / 2.0

        world_positions: list[Vector3] = []
        screen_positions: list[Vector2] = []

        for v in mesh.vertices:
            world_point = model.transform_point(v)
            wx: float = world_point[0]
            wy: float = world_point[1]
            wz: float = world_point[2]
            world_positions.append(Vector3(wx, wy, wz))

            clip = mvp.transform_point(v)
            cw: float = clip[3]
            ndc_x: float = clip[0] / cw
            ndc_y: float = clip[1] / cw
            sx: int = int(half_w + ndc_x * half_w)
            sy: int = int(half_h - ndc_y * half_h)
            screen_positions.append(Vector2(sx, sy))

        visible_faces: list[tuple[int, int, int, int]] = []
        visible_colors: list[int] = []
        depths: list[float] = []

        shader_fn: Callable[[ShaderContext], int] = self.shader

        face_idx: int = 0
        while face_idx < len(mesh.faces):
            face = mesh.faces[face_idx]
            i0: int = face[0]
            i1: int = face[1]
            i2: int = face[2]
            i3: int = face[3]
            p0: Vector3 = world_positions[i0]
            p1: Vector3 = world_positions[i1]
            p2: Vector3 = world_positions[i2]
            p3: Vector3 = world_positions[i3]
            e1: Vector3 = p1 - p0
            e2: Vector3 = p2 - p0
            n: Vector3 = e1.cross(e2)
            normal: Vector3 = n.normalized()

            to_camera: Vector3 = self.camera.position - p0
            if normal.dot(to_camera) > 0.0:
                cx: float = (p0.x + p1.x + p2.x + p3.x) / 4.0
                cy: float = (p0.y + p1.y + p2.y + p3.y) / 4.0
                cz: float = (p0.z + p1.z + p2.z + p3.z) / 4.0
                centroid: Vector3 = Vector3(cx, cy, cz)

                l0: Vector3 = mesh.vertices[i0]
                l1: Vector3 = mesh.vertices[i1]
                l2: Vector3 = mesh.vertices[i2]
                l3: Vector3 = mesh.vertices[i3]
                lcx: float = (l0.x + l1.x + l2.x + l3.x) / 4.0
                lcy: float = (l0.y + l1.y + l2.y + l3.y) / 4.0
                lcz: float = (l0.z + l1.z + l2.z + l3.z) / 4.0
                local_centroid: Vector3 = Vector3(lcx, lcy, lcz)

                ctx = ShaderContext(
                    normal,
                    centroid,
                    local_centroid,
                    self.camera.position,
                    self.light_dir,
                    self.light_color,
                    self.ambient,
                    self.specular,
                    self.shininess,
                    colors[face_idx],
                    self.no_texture,
                    0.5,
                    0.5,
                    0,
                    0,
                )
                shaded_color: int = shader_fn(ctx)

                to_cam_x: float = self.camera.position.x - cx
                to_cam_y: float = self.camera.position.y - cy
                to_cam_z: float = self.camera.position.z - cz
                dist_sq: float = (
                    to_cam_x * to_cam_x + to_cam_y * to_cam_y + to_cam_z * to_cam_z
                )

                visible_faces.append(face)
                visible_colors.append(shaded_color)
                depths.append(-dist_sq)
            face_idx = face_idx + 1

        order: list[int] = []
        oidx: int = 0
        while oidx < len(visible_faces):
            order.append(oidx)
            oidx = oidx + 1

        oi: int = 1
        while oi < len(order):
            cur: int = order[oi]
            cur_depth: float = depths[cur]
            oj: int = oi - 1
            while oj >= 0 and depths[order[oj]] > cur_depth:
                order[oj + 1] = order[oj]
                oj = oj - 1
            order[oj + 1] = cur
            oi = oi + 1

        for idx in order:
            face = visible_faces[idx]
            i0: int = face[0]
            i1: int = face[1]
            i2: int = face[2]
            i3: int = face[3]
            quad = PointShape([
                screen_positions[i0],
                screen_positions[i1],
                screen_positions[i2],
                screen_positions[i3],
            ])
            self.window.draw(quad, visible_colors[idx])

    def render_solid_per_pixel(
        self,
        mesh: Mesh,
        model: Matrix4,
        colors: list[int],
        texture: Texture,
    ):
        """Like render_solid(), but self.pixel_shader is called once per
        pixel inside each visible face (with ShaderContext.world_pos and
        .screen_x/.screen_y interpolated across the face) instead of once
        per face. Needed for effects that vary across a single flat quad:
        procedural patterns, screen-space effects, distance-based fog
        evaluated per-pixel rather than from one averaged centroid.

        normal is still constant per face (Mesh.cube()'s faces are flat),
        so this doesn't add smooth/Gouraud-style normal interpolation --
        only world_pos, screen position, and texture UV vary per pixel.

        texture is the one Texture sampled by textured_shader; pass any
        placeholder (e.g. Texture.solid(1, 1, 0)) when self.pixel_shader
        doesn't use ctx.texture at all.
        """
        proj: Matrix4 = self.camera.projection_matrix()
        view: Matrix4 = self.camera.view_matrix()
        vp: Matrix4 = proj.multiply(view)
        mvp: Matrix4 = vp.multiply(model)

        half_w: float = self.window.width / 2.0
        half_h: float = self.window.height / 2.0

        world_positions: list[Vector3] = []
        screen_positions: list[Vector2] = []
        inv_w_list: list[float] = []

        for v in mesh.vertices:
            world_point = model.transform_point(v)
            wx: float = world_point[0]
            wy: float = world_point[1]
            wz: float = world_point[2]
            world_positions.append(Vector3(wx, wy, wz))

            clip = mvp.transform_point(v)
            cw: float = clip[3]
            ndc_x: float = clip[0] / cw
            ndc_y: float = clip[1] / cw
            sx: int = int(half_w + ndc_x * half_w)
            sy: int = int(half_h - ndc_y * half_h)
            screen_positions.append(Vector2(sx, sy))
            inv_w_list.append(1.0 / cw)

        # World-space per-vertex normal, for Gouraud (smooth) shading --
        # transform_direction() rather than transform_point() since a
        # normal is a direction, not a position (no translation applied).
        world_normals: list[Vector3] = []
        for vn in mesh.vertex_normals:
            wn = model.transform_direction(vn)
            world_normals.append(Vector3(wn[0], wn[1], wn[2]))

        visible_faces: list[tuple[int, int, int, int]] = []
        visible_face_idx: list[int] = []
        visible_colors: list[int] = []
        depths: list[float] = []

        face_idx: int = 0
        while face_idx < len(mesh.faces):
            face = mesh.faces[face_idx]
            i0: int = face[0]
            i1: int = face[1]
            i2: int = face[2]
            i3: int = face[3]
            p0: Vector3 = world_positions[i0]
            p1: Vector3 = world_positions[i1]
            p2: Vector3 = world_positions[i2]
            p3: Vector3 = world_positions[i3]
            e1: Vector3 = p1 - p0
            e2: Vector3 = p2 - p0
            n: Vector3 = e1.cross(e2)
            normal: Vector3 = n.normalized()

            to_camera: Vector3 = self.camera.position - p0
            if normal.dot(to_camera) > 0.0:
                cx: float = (p0.x + p1.x + p2.x + p3.x) / 4.0
                cy: float = (p0.y + p1.y + p2.y + p3.y) / 4.0
                cz: float = (p0.z + p1.z + p2.z + p3.z) / 4.0
                to_cam_x: float = self.camera.position.x - cx
                to_cam_y: float = self.camera.position.y - cy
                to_cam_z: float = self.camera.position.z - cz
                dist_sq: float = (
                    to_cam_x * to_cam_x + to_cam_y * to_cam_y + to_cam_z * to_cam_z
                )

                visible_faces.append(face)
                visible_face_idx.append(face_idx)
                visible_colors.append(colors[face_idx])
                depths.append(-dist_sq)
            face_idx = face_idx + 1

        order: list[int] = []
        oidx: int = 0
        while oidx < len(visible_faces):
            order.append(oidx)
            oidx = oidx + 1

        oi: int = 1
        while oi < len(order):
            cur: int = order[oi]
            cur_depth: float = depths[cur]
            oj: int = oi - 1
            while oj >= 0 and depths[order[oj]] > cur_depth:
                order[oj + 1] = order[oj]
                oj = oj - 1
            order[oj + 1] = cur
            oi = oi + 1

        pixel_shader: Callable[[ShaderContext], int] = self.pixel_shader

        for idx in order:
            face = visible_faces[idx]
            i0: int = face[0]
            i1: int = face[1]
            i2: int = face[2]
            i3: int = face[3]
            quad_screen: list[Vector2] = [
                screen_positions[i0],
                screen_positions[i1],
                screen_positions[i2],
                screen_positions[i3],
            ]
            quad_world: list[Vector3] = [
                world_positions[i0],
                world_positions[i1],
                world_positions[i2],
                world_positions[i3],
            ]
            quad_local: list[Vector3] = [
                mesh.vertices[i0],
                mesh.vertices[i1],
                mesh.vertices[i2],
                mesh.vertices[i3],
            ]
            quad_normals: list[Vector3] = [
                world_normals[i0],
                world_normals[i1],
                world_normals[i2],
                world_normals[i3],
            ]
            quad_inv_w: list[float] = [
                inv_w_list[i0],
                inv_w_list[i1],
                inv_w_list[i2],
                inv_w_list[i3],
            ]
            orig_face_idx: int = visible_face_idx[idx]
            quad_u: list[float] = [
                mesh.face_u0[orig_face_idx],
                mesh.face_u1[orig_face_idx],
                mesh.face_u2[orig_face_idx],
                mesh.face_u3[orig_face_idx],
            ]
            quad_v: list[float] = [
                mesh.face_v0[orig_face_idx],
                mesh.face_v1[orig_face_idx],
                mesh.face_v2[orig_face_idx],
                mesh.face_v3[orig_face_idx],
            ]
            self._fill_quad_per_pixel(
                quad_screen,
                quad_world,
                quad_local,
                quad_inv_w,
                quad_u,
                quad_v,
                quad_normals,
                visible_colors[idx],
                texture,
                pixel_shader,
            )

    def _fill_quad_per_pixel(
        self,
        points: list[Vector2],
        world_points: list[Vector3],
        local_points: list[Vector3],
        inv_w: list[float],
        tex_us: list[float],
        tex_vs: list[float],
        vertex_normals: list[Vector3],
        base_color: int,
        texture: Texture,
        pixel_shader: Callable[[ShaderContext], int],
    ):
        """Scanline fill (same half-open edge rule as Window.draw()'s
        polygon path) that perspective-correctly interpolates world
        position, UV, and the per-vertex normal (Gouraud shading) across
        the face, calling pixel_shader once per pixel.

        Interpolating world_pos directly in screen space (rather than
        world_pos/w and 1/w, dividing back out per pixel) is only correct
        for an orthographic projection -- under perspective it visibly
        warps a regular pattern, the same "wobbly texture" artifact early
        3D hardware without perspective-correct texturing had.

        ShaderContext and its world_pos Vector3 are allocated once per
        face and mutated per pixel rather than reallocated, since a 640x480
        face can cover ~100k+ pixels and reallocating both per pixel was
        measurably slow.
        """
        min_y: int = points[0].y
        max_y: int = points[0].y
        k: int = 1
        while k < len(points):
            if points[k].y < min_y:
                min_y = points[k].y
            if points[k].y > max_y:
                max_y = points[k].y
            k = k + 1

        # world_pos/w, local_pos/w, normal/w, and tex_u/w, tex_v/w per
        # vertex, precomputed once.
        wow_x: list[float] = []
        wow_y: list[float] = []
        wow_z: list[float] = []
        low_x: list[float] = []
        low_y: list[float] = []
        low_z: list[float] = []
        nox: list[float] = []
        noy: list[float] = []
        noz: list[float] = []
        uow: list[float] = []
        vow: list[float] = []
        vi: int = 0
        while vi < len(world_points):
            wp: Vector3 = world_points[vi]
            lp: Vector3 = local_points[vi]
            vn: Vector3 = vertex_normals[vi]
            iw: float = inv_w[vi]
            wow_x.append(wp.x * iw)
            wow_y.append(wp.y * iw)
            wow_z.append(wp.z * iw)
            low_x.append(lp.x * iw)
            low_y.append(lp.y * iw)
            low_z.append(lp.z * iw)
            nox.append(vn.x * iw)
            noy.append(vn.y * iw)
            noz.append(vn.z * iw)
            uow.append(tex_us[vi] * iw)
            vow.append(tex_vs[vi] * iw)
            vi = vi + 1

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
            base_color,
            texture,
            0.0,
            0.0,
            0,
            0,
        )
        ctx_world_pos: Vector3 = ctx.world_pos
        ctx_local_pos: Vector3 = ctx.local_pos
        ctx_normal: Vector3 = ctx.normal

        for y in range(min_y, max_y + 1):
            xs: list[int] = []
            e_iw: list[float] = []
            e_wow_x: list[float] = []
            e_wow_y: list[float] = []
            e_wow_z: list[float] = []
            e_low_x: list[float] = []
            e_low_y: list[float] = []
            e_low_z: list[float] = []
            e_nox: list[float] = []
            e_noy: list[float] = []
            e_noz: list[float] = []
            e_uow: list[float] = []
            e_vow: list[float] = []

            i: int = 0
            while i < len(points):
                ni: int = (i + 1) % len(points)
                p1: Vector2 = points[i]
                p2: Vector2 = points[ni]

                if p1.y != p2.y:
                    edge_y_min: int = min(p1.y, p2.y)
                    edge_y_max: int = max(p1.y, p2.y)
                    if y == max_y:
                        in_range = edge_y_min <= y <= edge_y_max
                    else:
                        in_range = edge_y_min <= y < edge_y_max

                    if in_range:
                        numer: int = (y - p1.y) * (p2.x - p1.x)
                        den: int = p2.y - p1.y
                        x: int = p1.x + numer // den

                        t_num: float = float(y - p1.y)
                        t_den: float = float(p2.y - p1.y)
                        t: float = t_num / t_den

                        iw1: float = inv_w[i]
                        iw2: float = inv_w[ni]

                        xs.append(x)
                        e_iw.append(iw1 + (iw2 - iw1) * t)
                        e_wow_x.append(wow_x[i] + (wow_x[ni] - wow_x[i]) * t)
                        e_wow_y.append(wow_y[i] + (wow_y[ni] - wow_y[i]) * t)
                        e_wow_z.append(wow_z[i] + (wow_z[ni] - wow_z[i]) * t)
                        e_low_x.append(low_x[i] + (low_x[ni] - low_x[i]) * t)
                        e_low_y.append(low_y[i] + (low_y[ni] - low_y[i]) * t)
                        e_low_z.append(low_z[i] + (low_z[ni] - low_z[i]) * t)
                        e_nox.append(nox[i] + (nox[ni] - nox[i]) * t)
                        e_noy.append(noy[i] + (noy[ni] - noy[i]) * t)
                        e_noz.append(noz[i] + (noz[ni] - noz[i]) * t)
                        e_uow.append(uow[i] + (uow[ni] - uow[i]) * t)
                        e_vow.append(vow[i] + (vow[ni] - vow[i]) * t)
                i = i + 1

            n: int = len(xs)
            if n >= 2:
                si: int = 1
                while si < n:
                    cur_x: int = xs[si]
                    cur_iw: float = e_iw[si]
                    cur_wx: float = e_wow_x[si]
                    cur_wy: float = e_wow_y[si]
                    cur_wz: float = e_wow_z[si]
                    cur_lx: float = e_low_x[si]
                    cur_ly: float = e_low_y[si]
                    cur_lz: float = e_low_z[si]
                    cur_nx: float = e_nox[si]
                    cur_ny: float = e_noy[si]
                    cur_nz: float = e_noz[si]
                    cur_u: float = e_uow[si]
                    cur_v: float = e_vow[si]
                    sj: int = si - 1
                    while sj >= 0 and xs[sj] > cur_x:
                        xs[sj + 1] = xs[sj]
                        e_iw[sj + 1] = e_iw[sj]
                        e_wow_x[sj + 1] = e_wow_x[sj]
                        e_wow_y[sj + 1] = e_wow_y[sj]
                        e_wow_z[sj + 1] = e_wow_z[sj]
                        e_low_x[sj + 1] = e_low_x[sj]
                        e_low_y[sj + 1] = e_low_y[sj]
                        e_low_z[sj + 1] = e_low_z[sj]
                        e_nox[sj + 1] = e_nox[sj]
                        e_noy[sj + 1] = e_noy[sj]
                        e_noz[sj + 1] = e_noz[sj]
                        e_uow[sj + 1] = e_uow[sj]
                        e_vow[sj + 1] = e_vow[sj]
                        sj = sj - 1
                    xs[sj + 1] = cur_x
                    e_iw[sj + 1] = cur_iw
                    e_wow_x[sj + 1] = cur_wx
                    e_wow_y[sj + 1] = cur_wy
                    e_wow_z[sj + 1] = cur_wz
                    e_low_x[sj + 1] = cur_lx
                    e_low_y[sj + 1] = cur_ly
                    e_low_z[sj + 1] = cur_lz
                    e_nox[sj + 1] = cur_nx
                    e_noy[sj + 1] = cur_ny
                    e_noz[sj + 1] = cur_nz
                    e_uow[sj + 1] = cur_u
                    e_vow[sj + 1] = cur_v
                    si = si + 1

                j: int = 0
                while j + 1 < n:
                    x1: int = xs[j]
                    x2: int = xs[j + 1]
                    span: int = x2 - x1

                    iw_a: float = e_iw[j]
                    iw_b: float = e_iw[j + 1]
                    wx_a: float = e_wow_x[j]
                    wx_b: float = e_wow_x[j + 1]
                    wy_a: float = e_wow_y[j]
                    wy_b: float = e_wow_y[j + 1]
                    wz_a: float = e_wow_z[j]
                    wz_b: float = e_wow_z[j + 1]
                    lx_a: float = e_low_x[j]
                    lx_b: float = e_low_x[j + 1]
                    ly_a: float = e_low_y[j]
                    ly_b: float = e_low_y[j + 1]
                    lz_a: float = e_low_z[j]
                    lz_b: float = e_low_z[j + 1]
                    nx_a: float = e_nox[j]
                    nx_b: float = e_nox[j + 1]
                    ny_a: float = e_noy[j]
                    ny_b: float = e_noy[j + 1]
                    nz_a: float = e_noz[j]
                    nz_b: float = e_noz[j + 1]
                    uo_a: float = e_uow[j]
                    uo_b: float = e_uow[j + 1]
                    vo_a: float = e_vow[j]
                    vo_b: float = e_vow[j + 1]

                    px: int = x1
                    while px <= x2:
                        if span > 0:
                            u: float = float(px - x1) / float(span)
                        else:
                            u: float = 0.0

                        p_iw: float = iw_a + (iw_b - iw_a) * u
                        p_w: float = 1.0 / p_iw
                        p_wx: float = wx_a + (wx_b - wx_a) * u
                        p_wy: float = wy_a + (wy_b - wy_a) * u
                        p_wz: float = wz_a + (wz_b - wz_a) * u
                        p_lx: float = lx_a + (lx_b - lx_a) * u
                        p_ly: float = ly_a + (ly_b - ly_a) * u
                        p_lz: float = lz_a + (lz_b - lz_a) * u
                        p_nx: float = nx_a + (nx_b - nx_a) * u
                        p_ny: float = ny_a + (ny_b - ny_a) * u
                        p_nz: float = nz_a + (nz_b - nz_a) * u
                        p_u: float = uo_a + (uo_b - uo_a) * u
                        p_v: float = vo_a + (vo_b - vo_a) * u

                        ctx_world_pos.x = p_wx * p_w
                        ctx_world_pos.y = p_wy * p_w
                        ctx_world_pos.z = p_wz * p_w
                        ctx_local_pos.x = p_lx * p_w
                        ctx_local_pos.y = p_ly * p_w
                        ctx_local_pos.z = p_lz * p_w
                        # Linear interpolation (then dividing by w) doesn't
                        # preserve unit length, so the averaged normal must
                        # be renormalized per pixel -- otherwise lighting
                        # intensity (normal.dot(light_dir)) would be biased
                        # by how far |normal| has drifted from 1.
                        ctx_normal.x = p_nx * p_w
                        ctx_normal.y = p_ny * p_w
                        ctx_normal.z = p_nz * p_w
                        ctx_normal.normalize_in_place()
                        ctx.tex_u = p_u * p_w
                        ctx.tex_v = p_v * p_w
                        ctx.screen_x = px
                        ctx.screen_y = y
                        self.window.put_pixel(px, y, pixel_shader(ctx))
                        px = px + 1
                    j = j + 2

    def render_scene(self, objects: list[SceneObject]):
        """Like render_solid_per_pixel(), but depth-sorts visible faces
        across every object in the scene together, instead of each
        object's faces only being sorted against its own other faces and
        then drawn object-by-object (which would let a farther object's
        face wrongly cover a nearer object's face just because it was
        drawn in a later self.window.draw() call).

        Pass 1 below only computes each face's world-space centroid depth
        (cheap: 4 points) to build one global draw order across all
        objects. Pass 2 redoes the full per-vertex transform (screen
        position, world/local position, vertex normal, inv_w) but only
        for the 4 vertices of each face actually being drawn, in that
        global order -- a face's full vertex data isn't kept around
        between the two passes (it would otherwise need one array of
        per-object per-vertex data per object, i.e. a nested list, an
        untested-and-likely-fragile shape in asmpython elsewhere in this
        project), so it's cheaply recomputed once per visible face instead.
        """
        proj: Matrix4 = self.camera.projection_matrix()
        view: Matrix4 = self.camera.view_matrix()
        vp: Matrix4 = proj.multiply(view)

        global_depths: list[float] = []
        global_obj_idx: list[int] = []
        global_face_idx: list[int] = []

        oi: int = 0
        while oi < len(objects):
            obj: SceneObject = objects[oi]
            mesh: Mesh = obj.mesh
            model: Matrix4 = obj.model

            world_positions: list[Vector3] = []
            for v in mesh.vertices:
                wp = model.transform_point(v)
                world_positions.append(Vector3(wp[0], wp[1], wp[2]))

            fi: int = 0
            while fi < len(mesh.faces):
                face = mesh.faces[fi]
                i0: int = face[0]
                i1: int = face[1]
                i2: int = face[2]
                i3: int = face[3]
                p0: Vector3 = world_positions[i0]
                p1: Vector3 = world_positions[i1]
                p2: Vector3 = world_positions[i2]
                p3: Vector3 = world_positions[i3]
                e1: Vector3 = p1 - p0
                e2: Vector3 = p2 - p0
                n: Vector3 = e1.cross(e2)
                normal: Vector3 = n.normalized()

                to_camera: Vector3 = self.camera.position - p0
                if normal.dot(to_camera) > 0.0:
                    cx: float = (p0.x + p1.x + p2.x + p3.x) / 4.0
                    cy: float = (p0.y + p1.y + p2.y + p3.y) / 4.0
                    cz: float = (p0.z + p1.z + p2.z + p3.z) / 4.0
                    to_cam_x: float = self.camera.position.x - cx
                    to_cam_y: float = self.camera.position.y - cy
                    to_cam_z: float = self.camera.position.z - cz
                    dist_sq: float = (
                        to_cam_x * to_cam_x + to_cam_y * to_cam_y + to_cam_z * to_cam_z
                    )

                    global_depths.append(-dist_sq)
                    global_obj_idx.append(oi)
                    global_face_idx.append(fi)
                fi = fi + 1
            oi = oi + 1

        order: list[int] = []
        oidx: int = 0
        while oidx < len(global_depths):
            order.append(oidx)
            oidx = oidx + 1

        si: int = 1
        while si < len(order):
            cur: int = order[si]
            cur_depth: float = global_depths[cur]
            sj: int = si - 1
            while sj >= 0 and global_depths[order[sj]] > cur_depth:
                order[sj + 1] = order[sj]
                sj = sj - 1
            order[sj + 1] = cur
            si = si + 1

        half_w: float = self.window.width / 2.0
        half_h: float = self.window.height / 2.0
        pixel_shader: Callable[[ShaderContext], int] = self.pixel_shader

        for idx in order:
            obj_idx: int = global_obj_idx[idx]
            face_idx: int = global_face_idx[idx]
            obj: SceneObject = objects[obj_idx]
            mesh: Mesh = obj.mesh
            model: Matrix4 = obj.model
            mvp: Matrix4 = vp.multiply(model)

            face = mesh.faces[face_idx]
            face_vertex_idx: list[int] = [face[0], face[1], face[2], face[3]]

            quad_screen: list[Vector2] = []
            quad_world: list[Vector3] = []
            quad_local: list[Vector3] = []
            quad_normals: list[Vector3] = []
            quad_inv_w: list[float] = []

            vi: int = 0
            while vi < 4:
                vidx: int = face_vertex_idx[vi]
                v: Vector3 = mesh.vertices[vidx]

                world_point = model.transform_point(v)
                quad_world.append(Vector3(world_point[0], world_point[1], world_point[2]))
                quad_local.append(v)

                clip = mvp.transform_point(v)
                cw: float = clip[3]
                ndc_x: float = clip[0] / cw
                ndc_y: float = clip[1] / cw
                sx: int = int(half_w + ndc_x * half_w)
                sy: int = int(half_h - ndc_y * half_h)
                quad_screen.append(Vector2(sx, sy))
                quad_inv_w.append(1.0 / cw)

                vn: Vector3 = mesh.vertex_normals[vidx]
                wn = model.transform_direction(vn)
                quad_normals.append(Vector3(wn[0], wn[1], wn[2]))
                vi = vi + 1

            quad_u: list[float] = [
                mesh.face_u0[face_idx],
                mesh.face_u1[face_idx],
                mesh.face_u2[face_idx],
                mesh.face_u3[face_idx],
            ]
            quad_v: list[float] = [
                mesh.face_v0[face_idx],
                mesh.face_v1[face_idx],
                mesh.face_v2[face_idx],
                mesh.face_v3[face_idx],
            ]

            self._fill_quad_per_pixel(
                quad_screen,
                quad_world,
                quad_local,
                quad_inv_w,
                quad_u,
                quad_v,
                quad_normals,
                obj.colors[face_idx],
                obj.texture,
                pixel_shader,
            )
