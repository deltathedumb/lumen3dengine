from ._vector import Vector3


class Mesh:
    vertices: list[Vector3]
    edges: list[tuple[int, int]]
    # Every face is a triangle (3 vertex indices), wound so cross(v1-v0,
    # v2-v0) points outward -- this is what backface culling checks. A quad
    # source shape (e.g. a cube face) is split into 2 triangles sharing a
    # diagonal; this is the universal primitive every renderer path (and
    # every loaded mesh format -- OBJ, glTF, ...) ultimately reduces to, so
    # the renderer only ever needs to know how to rasterize one shape.
    triangles: list[tuple[int, int, int]]
    # UV coords for each triangle's 3 corners, parallel to `triangles`
    # (tri_u0[i], tri_v0[i] is the UV for triangles[i][0], and so on). Flat
    # parallel float lists rather than nested per-triangle lists/tuples,
    # matching every other per-face list in the renderer (depths, colors,
    # ...) -- nested list[list[...]] shapes are an untested/fragile path in
    # asmpython elsewhere in this project (see Renderer3D.render_scene).
    #
    # These can't just live on `vertices` alongside position, since the same
    # shared vertex (a cube corner is shared by 3 faces) generally needs a
    # *different* UV per face it's part of -- a per-vertex UV would force
    # every face touching that corner to agree on one mapping.
    tri_u0: list[float]
    tri_v0: list[float]
    tri_u1: list[float]
    tri_v1: list[float]
    tri_u2: list[float]
    tri_v2: list[float]
    # One normal per vertex (local space, unit length), parallel to
    # `vertices` -- the average of every adjacent triangle's flat normal,
    # area-weighted by the triangle's own cross-product magnitude before
    # normalizing (a small sliver triangle shouldn't sway a vertex normal as
    # much as a large one meeting at the same vertex).
    # Renderer3D's per-pixel path interpolates these across each triangle
    # (Gouraud shading) instead of using one flat per-triangle normal, for
    # smooth-looking lighting on a mesh that's meant to look rounded.
    vertex_normals: list[Vector3]
    # GPU upload cache for GLRenderer3D (see pugtk/_renderer3d_gl.py).
    # Unused by the software Renderer3D -- gl_uploaded stays 0 for any
    # Mesh that's never drawn through GLRenderer3D, and these fields cost
    # nothing beyond the per-instance storage. Caching here (rather than a
    # separate id(mesh)-keyed dict inside GLRenderer3D) means re-rendering
    # the same Mesh next frame is a single `if mesh.gl_uploaded` check, not
    # a dict lookup, and the cache can never go stale by outliving its
    # Mesh (it's freed whenever the Mesh itself is).
    gl_uploaded: int
    gl_vao: int
    gl_vbo: int
    gl_vertex_count: int

    def __init__(
        self,
        vertices: list[Vector3],
        edges: list[tuple[int, int]],
        triangles: list[tuple[int, int, int]],
        tri_u0: list[float],
        tri_v0: list[float],
        tri_u1: list[float],
        tri_v1: list[float],
        tri_u2: list[float],
        tri_v2: list[float],
        vertex_normals: list[Vector3],
    ):
        self.vertices = vertices
        self.edges = edges
        self.triangles = triangles
        self.tri_u0 = tri_u0
        self.tri_v0 = tri_v0
        self.tri_u1 = tri_u1
        self.tri_v1 = tri_v1
        self.tri_u2 = tri_u2
        self.tri_v2 = tri_v2
        self.vertex_normals = vertex_normals
        self.gl_uploaded = 0
        self.gl_vao = 0
        self.gl_vbo = 0
        self.gl_vertex_count = 0

    @staticmethod
    def compute_vertex_normals(
        vertices: list[Vector3],
        triangles: list[tuple[int, int, int]],
    ) -> list[Vector3]:
        """Averages each triangle's flat normal (weighted by its own
        unnormalized cross-product length, i.e. twice its area) into its 3
        vertices, then normalizes. A vertex shared by triangles that aren't
        coplanar (e.g. a cube corner) ends up with a normal pointing
        diagonally between them -- exactly the "rounded corner" look
        Gouraud shading is known for, not a bug."""
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

        ti: int = 0
        while ti < len(triangles):
            tri = triangles[ti]
            i0: int = tri[0]
            i1: int = tri[1]
            i2: int = tri[2]
            p0: Vector3 = vertices[i0]
            p1: Vector3 = vertices[i1]
            p2: Vector3 = vertices[i2]
            e1: Vector3 = p1 - p0
            e2: Vector3 = p2 - p0
            fn: Vector3 = e1.cross(e2)

            sum_x[i0] = sum_x[i0] + fn.x
            sum_y[i0] = sum_y[i0] + fn.y
            sum_z[i0] = sum_z[i0] + fn.z
            sum_x[i1] = sum_x[i1] + fn.x
            sum_y[i1] = sum_y[i1] + fn.y
            sum_z[i1] = sum_z[i1] + fn.z
            sum_x[i2] = sum_x[i2] + fn.x
            sum_y[i2] = sum_y[i2] + fn.y
            sum_z[i2] = sum_z[i2] + fn.z
            ti = ti + 1

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
        # Each face quad (0,1,2,3 corners, outward-wound) split into 2
        # triangles (0,1,2) and (0,2,3) sharing the 0-2 diagonal. UVs follow
        # the same (0,0)/(1,0)/(1,1)/(0,1) corner mapping as the old
        # per-face quad UVs, just distributed across the 2 triangles.
        quads: list[tuple[int, int, int, int]] = [
            (0, 3, 2, 1),  # back (z = -h)
            (4, 5, 6, 7),  # front (z = +h)
            (0, 4, 7, 3),  # left (x = -h)
            (1, 2, 6, 5),  # right (x = +h)
            (0, 1, 5, 4),  # bottom (y = -h)
            (3, 7, 6, 2),  # top (y = +h)
        ]

        triangles: list[tuple[int, int, int]] = []
        tri_u0: list[float] = []
        tri_v0: list[float] = []
        tri_u1: list[float] = []
        tri_v1: list[float] = []
        tri_u2: list[float] = []
        tri_v2: list[float] = []

        qi: int = 0
        while qi < len(quads):
            quad = quads[qi]
            a: int = quad[0]
            b: int = quad[1]
            c: int = quad[2]
            d: int = quad[3]

            triangles.append((a, b, c))
            tri_u0.append(0.0)
            tri_v0.append(0.0)
            tri_u1.append(1.0)
            tri_v1.append(0.0)
            tri_u2.append(1.0)
            tri_v2.append(1.0)

            triangles.append((a, c, d))
            tri_u0.append(0.0)
            tri_v0.append(0.0)
            tri_u1.append(1.0)
            tri_v1.append(1.0)
            tri_u2.append(0.0)
            tri_v2.append(1.0)

            qi = qi + 1

        vertex_normals: list[Vector3] = Mesh.compute_vertex_normals(vertices, triangles)
        return Mesh(
            vertices,
            edges,
            triangles,
            tri_u0,
            tri_v0,
            tri_u1,
            tri_v1,
            tri_u2,
            tri_v2,
            vertex_normals,
        )
