from ._vector import Vector3


class Mesh:
    vertices: list[Vector3]
    edges: list[tuple[int, int]]
    # Each face is a planar, convex, consistently-wound quad (4 vertex
    # indices). Earlier this was 2 independently-culled triangles per quad,
    # but two coplanar triangles can get tiny floating-point differences in
    # their computed normals, so right at a silhouette angle one triangle
    # could cull while its partner doesn't -- tearing the face diagonally in
    # half. One normal/visibility decision per quad avoids that, and
    # Window.draw()'s scanline fill already handles arbitrary convex N-gons,
    # so the quad is drawn directly with no triangulation needed.
    faces: list[tuple[int, int, int, int]]
    # UV coords for each face's 4 corners, parallel to `faces` (face_u0[i],
    # face_v0[i] is the UV for faces[i][0], and so on). Kept as 8 separate
    # flat float lists rather than nested per-face lists/tuples, since
    # they're indexed the same way as every other per-face float list in
    # the renderer (depths, colors, etc).
    #
    # These can't just live on `vertices` alongside position, since the
    # same shared vertex (a cube corner is shared by 3 faces) generally
    # needs a *different* UV per face it's part of -- a per-vertex UV
    # would force every face touching that corner to agree on one mapping.
    face_u0: list[float]
    face_v0: list[float]
    face_u1: list[float]
    face_v1: list[float]
    face_u2: list[float]
    face_v2: list[float]
    face_u3: list[float]
    face_v3: list[float]
    # One normal per vertex (local space, unit length), parallel to
    # `vertices` -- the average of every adjacent face's flat normal.
    # Renderer3D.render_solid_per_pixel() interpolates these across each
    # face (Gouraud shading) instead of using one flat per-face normal, for
    # smooth-looking lighting on a mesh that's meant to look rounded.
    # render_solid() (one shader call per face) has no use for these; it
    # always shades with the flat per-face normal.
    vertex_normals: list[Vector3]

    def __init__(
        self,
        vertices: list[Vector3],
        edges: list[tuple[int, int]],
        faces: list[tuple[int, int, int, int]],
        face_u0: list[float],
        face_v0: list[float],
        face_u1: list[float],
        face_v1: list[float],
        face_u2: list[float],
        face_v2: list[float],
        face_u3: list[float],
        face_v3: list[float],
        vertex_normals: list[Vector3],
    ):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.face_u0 = face_u0
        self.face_v0 = face_v0
        self.face_u1 = face_u1
        self.face_v1 = face_v1
        self.face_u2 = face_u2
        self.face_v2 = face_v2
        self.face_u3 = face_u3
        self.face_v3 = face_v3
        self.vertex_normals = vertex_normals

    @staticmethod
    def compute_vertex_normals(
        vertices: list[Vector3],
        faces: list[tuple[int, int, int, int]],
    ) -> list[Vector3]:
        """Averages each face's flat normal into its 4 vertices, then
        normalizes. A vertex shared by faces that aren't coplanar (e.g. a
        cube corner, shared by 3 mutually perpendicular faces) ends up with
        a normal pointing diagonally between them -- exactly the
        "rounded corner" look Gouraud shading is known for, not a bug."""
        n: int = len(vertices)
        sum_x: list[float] = []
        sum_y: list[float] = []
        sum_z: list[float] = []
        i: int = 0
        while i < n:
            sum_x.append(0.0)
            sum_y.append(0.0)
            sum_z.append(0.0)
            i = i + 1

        fi: int = 0
        while fi < len(faces):
            face = faces[fi]
            i0: int = face[0]
            i1: int = face[1]
            i2: int = face[2]
            i3: int = face[3]
            p0: Vector3 = vertices[i0]
            p1: Vector3 = vertices[i1]
            p2: Vector3 = vertices[i2]
            e1: Vector3 = p1 - p0
            e2: Vector3 = p2 - p0
            face_normal: Vector3 = e1.cross(e2)
            fn: Vector3 = face_normal.normalized()

            sum_x[i0] = sum_x[i0] + fn.x
            sum_y[i0] = sum_y[i0] + fn.y
            sum_z[i0] = sum_z[i0] + fn.z
            sum_x[i1] = sum_x[i1] + fn.x
            sum_y[i1] = sum_y[i1] + fn.y
            sum_z[i1] = sum_z[i1] + fn.z
            sum_x[i2] = sum_x[i2] + fn.x
            sum_y[i2] = sum_y[i2] + fn.y
            sum_z[i2] = sum_z[i2] + fn.z
            sum_x[i3] = sum_x[i3] + fn.x
            sum_y[i3] = sum_y[i3] + fn.y
            sum_z[i3] = sum_z[i3] + fn.z
            fi = fi + 1

        result: list[Vector3] = []
        i = 0
        while i < n:
            accum: Vector3 = Vector3(sum_x[i], sum_y[i], sum_z[i])
            result.append(accum.normalized())
            i = i + 1
        return result

    @staticmethod
    def cube(size: float):
        h: float = size / 2.0
        vertices: list[Vector3] = [
            Vector3(-h, -h, -h),
            Vector3(h, -h, -h),
            Vector3(h, h, -h),
            Vector3(-h, h, -h),
            Vector3(-h, -h, h),
            Vector3(h, -h, h),
            Vector3(h, h, h),
            Vector3(-h, h, h),
        ]
        edges: list[tuple[int, int]] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        # Each quad wound so cross(v1-v0, v2-v0) points outward (verified by
        # hand for each face) -- this is what backface culling checks.
        faces: list[tuple[int, int, int, int]] = [
            (0, 3, 2, 1),  # back (z = -h)
            (4, 5, 6, 7),  # front (z = +h)
            (0, 4, 7, 3),  # left (x = -h)
            (1, 2, 6, 5),  # right (x = +h)
            (0, 1, 5, 4),  # bottom (y = -h)
            (3, 7, 6, 2),  # top (y = +h)
        ]
        # Same (0,0)/(1,0)/(1,1)/(0,1) corner mapping on every face -- the
        # whole texture tiled identically onto each of the 6 faces. A mesh
        # wanting a per-face atlas/different UVs just needs its own
        # face_u0..face_v3 lists built directly via the Mesh constructor.
        face_u0: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        face_v0: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        face_u1: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        face_v1: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        face_u2: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        face_v2: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        face_u3: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        face_v3: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        vertex_normals: list[Vector3] = Mesh.compute_vertex_normals(vertices, faces)
        return Mesh(
            vertices,
            edges,
            faces,
            face_u0,
            face_v0,
            face_u1,
            face_v1,
            face_u2,
            face_v2,
            face_u3,
            face_v3,
            vertex_normals,
        )
