"""Mesh.from_obj()/from_gltf() -- loading real exported assets (Blender,
Maya, etc.) instead of only pugtk's procedural Mesh.cube()/plane().

Split out from _mesh.py (which stays focused on the core Mesh data shape
and procedural constructors) since loading needs file I/O and text/binary
parsing the procedural constructors never touch.

Both loaders go through io.FileIO (pugtk/_renderer3d_gl.py's GL texture
upload established this same pattern: bypass the builtin open()/file
object, which is a no-op stub at the codegen level -- see asmpython's
CHANGELOG -- and use io.FileIO directly instead, which works today since
it's ordinary user-class instantiation/method dispatch, ths builtin
open()'s broken "any"-typed special-casing never enters into it).
"""
from __future__ import annotations

from io import FileIO

from ._vector import Vector3
from ._mesh import Mesh


def _tokenize_ws(line: str) -> list[str]:
    """Splits on runs of whitespace, dropping empty tokens -- str.split(" ")
    yields an empty string for every run of 2+ consecutive spaces (real
    Python str.split(sep) semantics: a separator argument means "split at
    each occurrence", not "split at each run"), which OBJ files routinely
    have (hand-edited/exported with aligned columns). This is the
    sep-less split() behavior (split-on-any-whitespace-run,
    drop-empties) that real Python's str.split() defaults to without an
    argument -- asmpython's split() always takes an explicit separator,
    so this rebuilds that default behavior on top of it."""
    raw: list = line.split(" ")
    result: list = []
    i: int = 0
    while i < len(raw):
        tok: str = raw[i]
        if len(tok) > 0:
            result.append(tok)
        i = i + 1
    return result


def _parse_obj_index(tok: str, count: int) -> int:
    """OBJ face-vertex indices are 1-based; a negative index counts back
    from the end of the vertex list seen so far (count). Returns a 0-based
    index into Mesh.vertices."""
    n: int = int(tok)
    if n < 0:
        return count + n
    return n - 1


def load_obj(path: str) -> Mesh:
    """Loads a Wavefront .obj file into a Mesh. Supports the common
    subset real exporters produce: `v`/`vt`/`f` lines (triangles or
    convex polygons, fan-triangulated), `vn` lines are read but ignored
    -- Mesh.vertex_normals is always recomputed from triangle geometry
    via Mesh.compute_vertex_normals() (area-weighted averaging), matching
    every procedural Mesh constructor (Mesh.cube()/plane()) rather than
    introducing a second, inconsistent normal source. Unsupported
    directives (`mtllib`, `usemtl`, `g`, `s`, `o`, ...) are silently
    skipped -- materials/grouping aren't part of Mesh's data model yet.
    """
    f = FileIO(path, "r")
    text: str = f.read()
    f.close()

    positions: list[Vector3] = []
    uvs_u: list[float] = []
    uvs_v: list[float] = []
    triangles: list = []
    tri_u0: list[float] = []
    tri_v0: list[float] = []
    tri_u1: list[float] = []
    tri_v1: list[float] = []
    tri_u2: list[float] = []
    tri_v2: list[float] = []

    lines: list = text.splitlines()
    li: int = 0
    while li < len(lines):
        line: str = lines[li].strip()
        li = li + 1
        if len(line) == 0 or line.startswith("#"):
            continue
        parts: list = _tokenize_ws(line)
        if len(parts) == 0:
            continue
        tag: str = parts[0]

        if tag == "v":
            x: float = float(parts[1])
            y: float = float(parts[2])
            z: float = float(parts[3])
            positions.append(Vector3(x, y, z))
        elif tag == "vt":
            u: float = float(parts[1])
            v: float = float(parts[2])
            uvs_u.append(u)
            uvs_v.append(v)
        elif tag == "f":
            # parts[1:] are "v", "v/vt", "v//vn", or "v/vt/vn" tokens, one
            # per polygon corner -- fan-triangulate (corner0, corner_i,
            # corner_i+1) for i in 1..n-2, which is exact for the convex
            # polygons every real exporter emits (and exactly reproduces
            # the already-triangle case, n==3, as the single triangle
            # (0,1,2)).
            corner_v: list[int] = []
            corner_u: list[float] = []
            corner_w: list[float] = []
            ci: int = 1
            while ci < len(parts):
                corner: str = parts[ci]
                comps: list = corner.split("/")
                vi: int = _parse_obj_index(comps[0], len(positions))
                corner_v.append(vi)
                if len(comps) >= 2 and len(comps[1]) > 0:
                    ti: int = _parse_obj_index(comps[1], len(uvs_u))
                    corner_u.append(uvs_u[ti])
                    corner_w.append(uvs_v[ti])
                else:
                    corner_u.append(0.0)
                    corner_w.append(0.0)
                ci = ci + 1

            fi: int = 1
            while fi < len(corner_v) - 1:
                triangles.append((corner_v[0], corner_v[fi], corner_v[fi + 1]))
                tri_u0.append(corner_u[0])
                tri_v0.append(corner_w[0])
                tri_u1.append(corner_u[fi])
                tri_v1.append(corner_w[fi])
                tri_u2.append(corner_u[fi + 1])
                tri_v2.append(corner_w[fi + 1])
                fi = fi + 1
        # vn, mtllib, usemtl, g, s, o, l: skipped -- see docstring.

    edges: list = []
    vertex_normals: list[Vector3] = Mesh.compute_vertex_normals(positions, triangles)
    return Mesh(
        positions,
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
