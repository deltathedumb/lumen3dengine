"""Mesh.from_gltf() -- loads a glTF 2.0 (.gltf, JSON + embedded-base64 or
external .bin buffers) asset into pugtk's Mesh.

Supports the common single-mesh-single-primitive-triangles case real
exporters (Blender's "glTF Embedded"/"glTF Separate") produce:
POSITION/NORMAL/TEXCOORD_0 accessors, float32 components, an optional
index accessor (unsigned byte/short/int), TRIANGLES draw mode. Multi-
primitive meshes use only the first primitive; multi-mesh documents use
only meshes[0] -- a full scene-graph importer (nodes/transforms/multiple
meshes merged or kept separate) is future work, not this pass's scope.

Binary .glb (the single-file binary container format, JSON chunk +
binary chunk) is not yet supported -- only .gltf (JSON text, with buffers
either embedded as base64 data URIs or referencing an external .bin file
relative to the .gltf's own directory).
"""
from __future__ import annotations

import base64

from io import FileIO

from ._vector import Vector3
from ._mesh import Mesh
from ._gltf_json import (
    get_top_level_array_items,
    get_int_field,
    get_str_field,
    get_object_field,
)


_COMPONENT_TYPE_BYTE: int = 5120
_COMPONENT_TYPE_UNSIGNED_BYTE: int = 5121
_COMPONENT_TYPE_SHORT: int = 5122
_COMPONENT_TYPE_UNSIGNED_SHORT: int = 5123
_COMPONENT_TYPE_UNSIGNED_INT: int = 5125
_COMPONENT_TYPE_FLOAT: int = 5126


def _str_to_ascii_bytes(s: str) -> list[int]:
    result: list[int] = []
    i: int = 0
    n: int = len(s)
    while i < n:
        result.append(ord(s[i]))
        i = i + 1
    return result


def _decode_buffer_uri(uri: str, gltf_dir: str) -> list[int]:
    """A glTF buffer's `uri` is either a `data:...;base64,<data>` URI
    (the whole buffer embedded directly in the JSON -- what "glTF
    Embedded" exports produce) or a relative path to an external file
    (what "glTF Separate" exports produce, alongside one or more .bin
    files next to the .gltf itself)."""
    prefix: str = "data:"
    if len(uri) >= len(prefix) and uri[0:len(prefix)] == prefix:
        comma: int = uri.find(",")
        b64_text: str = uri[comma + 1:]
        ascii_bytes: list[int] = _str_to_ascii_bytes(b64_text)
        return base64.b64decode(ascii_bytes)
    path: str = gltf_dir + uri
    f = FileIO(path, "rb")
    data: list[int] = f.read_bytes()
    f.close()
    return data


def _bytes_to_float32(data: list[int], byte_offset: int) -> float:
    """Decodes 4 little-endian bytes at byte_offset into a Python float
    via IEEE-754 bit manipulation -- struct.unpack is available but this
    avoids a dependency on yet another stdlib module's own correctness
    for a single, simple, hot-path operation (called once per float
    component per vertex)."""
    b0: int = data[byte_offset]
    b1: int = data[byte_offset + 1]
    b2: int = data[byte_offset + 2]
    b3: int = data[byte_offset + 3]
    bits: int = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    sign: int = -1 if (bits >> 31) & 1 else 1
    exponent: int = (bits >> 23) & 0xFF
    mantissa: int = bits & 0x7FFFFF
    if exponent == 0 and mantissa == 0:
        return 0.0
    if exponent == 0:
        # Subnormal -- negligible for mesh data in practice, but handle
        # correctly rather than silently rounding to 0.
        m: float = float(mantissa) / float(1 << 23)
        return float(sign) * m * (2.0 ** -126)
    m2: float = 1.0 + float(mantissa) / float(1 << 23)
    e: int = exponent - 127
    return float(sign) * m2 * (2.0 ** e)


def _bytes_to_uint(data: list[int], byte_offset: int, component_type: int) -> int:
    if component_type == _COMPONENT_TYPE_UNSIGNED_BYTE or component_type == _COMPONENT_TYPE_BYTE:
        return data[byte_offset]
    if component_type == _COMPONENT_TYPE_UNSIGNED_SHORT or component_type == _COMPONENT_TYPE_SHORT:
        return data[byte_offset] | (data[byte_offset + 1] << 8)
    return (
        data[byte_offset]
        | (data[byte_offset + 1] << 8)
        | (data[byte_offset + 2] << 16)
        | (data[byte_offset + 3] << 24)
    )


def _component_byte_size(component_type: int) -> int:
    if component_type == _COMPONENT_TYPE_BYTE or component_type == _COMPONENT_TYPE_UNSIGNED_BYTE:
        return 1
    if component_type == _COMPONENT_TYPE_SHORT or component_type == _COMPONENT_TYPE_UNSIGNED_SHORT:
        return 2
    return 4


def _type_component_count(ty: str) -> int:
    if ty == "SCALAR":
        return 1
    if ty == "VEC2":
        return 2
    if ty == "VEC3":
        return 3
    if ty == "VEC4":
        return 4
    return 1


class _Accessor:
    buffer_view: int
    byte_offset: int
    component_type: int
    count: int
    component_count: int

    def __init__(self, buffer_view: int, byte_offset: int, component_type: int, count: int, component_count: int):
        self.buffer_view = buffer_view
        self.byte_offset = byte_offset
        self.component_type = component_type
        self.count = count
        self.component_count = component_count


class _BufferView:
    buffer: int
    byte_offset: int
    byte_length: int
    byte_stride: int

    def __init__(self, buffer: int, byte_offset: int, byte_length: int, byte_stride: int):
        self.buffer = buffer
        self.byte_offset = byte_offset
        self.byte_length = byte_length
        self.byte_stride = byte_stride


def _read_float_accessor(
    accessor: _Accessor, buffer_view: _BufferView, raw: list[int]
) -> list[float]:
    """Reads accessor.count vectors of accessor.component_count floats
    each, returned as one flat list (vec3 POSITION -> 3 floats/entry,
    in order). Assumes tightly-packed (byteStride 0) float32 data, the
    overwhelmingly common case for a position/normal/UV accessor from a
    real exporter -- an interleaved (nonzero byteStride) accessor would
    need stride-aware reads instead of this single-tight-loop version."""
    base: int = buffer_view.byte_offset + accessor.byte_offset
    stride: int = buffer_view.byte_stride
    if stride == 0:
        stride = accessor.component_count * 4
    result: list[float] = []
    i: int = 0
    while i < accessor.count:
        entry_off: int = base + i * stride
        c: int = 0
        while c < accessor.component_count:
            result.append(_bytes_to_float32(raw, entry_off + c * 4))
            c = c + 1
        i = i + 1
    return result


def _read_index_accessor(
    accessor: _Accessor, buffer_view: _BufferView, raw: list[int]
) -> list[int]:
    base: int = buffer_view.byte_offset + accessor.byte_offset
    comp_size: int = _component_byte_size(accessor.component_type)
    stride: int = buffer_view.byte_stride
    if stride == 0:
        stride = comp_size
    result: list[int] = []
    i: int = 0
    while i < accessor.count:
        entry_off: int = base + i * stride
        result.append(_bytes_to_uint(raw, entry_off, accessor.component_type))
        i = i + 1
    return result


def _gltf_dir(path: str) -> str:
    slash: int = path.rfind("/")
    backslash: int = path.rfind("\\")
    cut: int = slash if slash > backslash else backslash
    if cut < 0:
        return ""
    return path[0:cut + 1]


def load_gltf(path: str) -> Mesh:
    f = FileIO(path, "r")
    text: str = f.read()
    f.close()
    gltf_dir: str = _gltf_dir(path)

    buffer_objs: list[str] = get_top_level_array_items(text, "buffers")
    buffers: list[list[int]] = []
    bi: int = 0
    while bi < len(buffer_objs):
        uri: str = get_str_field(buffer_objs[bi], "uri", "")
        buffers.append(_decode_buffer_uri(uri, gltf_dir))
        bi = bi + 1

    bufferview_objs: list[str] = get_top_level_array_items(text, "bufferViews")
    buffer_views: list[_BufferView] = []
    vi: int = 0
    while vi < len(bufferview_objs):
        bv_text: str = bufferview_objs[vi]
        buffer_views.append(_BufferView(
            get_int_field(bv_text, "buffer", 0),
            get_int_field(bv_text, "byteOffset", 0),
            get_int_field(bv_text, "byteLength", 0),
            get_int_field(bv_text, "byteStride", 0),
        ))
        vi = vi + 1

    accessor_objs: list[str] = get_top_level_array_items(text, "accessors")
    accessors: list[_Accessor] = []
    ai: int = 0
    while ai < len(accessor_objs):
        acc_text: str = accessor_objs[ai]
        ty: str = get_str_field(acc_text, "type", "SCALAR")
        accessors.append(_Accessor(
            get_int_field(acc_text, "bufferView", 0),
            get_int_field(acc_text, "byteOffset", 0),
            get_int_field(acc_text, "componentType", _COMPONENT_TYPE_FLOAT),
            get_int_field(acc_text, "count", 0),
            _type_component_count(ty),
        ))
        ai = ai + 1

    mesh_objs: list[str] = get_top_level_array_items(text, "meshes")
    if len(mesh_objs) == 0:
        raise ValueError("glTF document has no meshes")
    primitive_objs: list[str] = get_top_level_array_items(mesh_objs[0], "primitives")
    if len(primitive_objs) == 0:
        raise ValueError("glTF mesh has no primitives")
    prim_text: str = primitive_objs[0]

    attrs_text: str = get_object_field(prim_text, "attributes")
    pos_accessor_idx: int = get_int_field(attrs_text, "POSITION", -1)
    if pos_accessor_idx < 0:
        raise ValueError("glTF primitive has no POSITION attribute")
    # NORMAL is intentionally not read: Mesh.vertex_normals is always
    # recomputed from triangle geometry (Mesh.compute_vertex_normals,
    # area-weighted averaging), matching every other Mesh constructor
    # (Mesh.cube()/plane()/_mesh_loader.load_obj()) rather than
    # introducing a second, inconsistent normal source.
    uv_accessor_idx: int = get_int_field(attrs_text, "TEXCOORD_0", -1)
    index_accessor_idx: int = get_int_field(prim_text, "indices", -1)

    pos_accessor: _Accessor = accessors[pos_accessor_idx]
    pos_view: _BufferView = buffer_views[pos_accessor.buffer_view]
    pos_floats: list[float] = _read_float_accessor(pos_accessor, pos_view, buffers[pos_view.buffer])

    positions: list[Vector3] = []
    i: int = 0
    while i < pos_accessor.count:
        positions.append(Vector3(pos_floats[i * 3], pos_floats[i * 3 + 1], pos_floats[i * 3 + 2]))
        i = i + 1

    uv_u: list[float] = []
    uv_v: list[float] = []
    if uv_accessor_idx >= 0:
        uv_accessor: _Accessor = accessors[uv_accessor_idx]
        uv_view: _BufferView = buffer_views[uv_accessor.buffer_view]
        uv_floats: list[float] = _read_float_accessor(uv_accessor, uv_view, buffers[uv_view.buffer])
        j: int = 0
        while j < uv_accessor.count:
            uv_u.append(uv_floats[j * 2])
            uv_v.append(uv_floats[j * 2 + 1])
            j = j + 1

    indices: list[int] = []
    if index_accessor_idx >= 0:
        idx_accessor: _Accessor = accessors[index_accessor_idx]
        idx_view: _BufferView = buffer_views[idx_accessor.buffer_view]
        indices = _read_index_accessor(idx_accessor, idx_view, buffers[idx_view.buffer])
    else:
        k: int = 0
        while k < pos_accessor.count:
            indices.append(k)
            k = k + 1

    triangles: list = []
    tri_u0: list[float] = []
    tri_v0: list[float] = []
    tri_u1: list[float] = []
    tri_v1: list[float] = []
    tri_u2: list[float] = []
    tri_v2: list[float] = []
    has_uv: int = len(uv_u) > 0

    ti: int = 0
    while ti < len(indices) - 2:
        i0: int = indices[ti]
        i1: int = indices[ti + 1]
        i2: int = indices[ti + 2]
        triangles.append((i0, i1, i2))
        if has_uv:
            tri_u0.append(uv_u[i0])
            tri_v0.append(uv_v[i0])
            tri_u1.append(uv_u[i1])
            tri_v1.append(uv_v[i1])
            tri_u2.append(uv_u[i2])
            tri_v2.append(uv_v[i2])
        else:
            tri_u0.append(0.0)
            tri_v0.append(0.0)
            tri_u1.append(0.0)
            tri_v1.append(0.0)
            tri_u2.append(0.0)
            tri_v2.append(0.0)
        ti = ti + 3

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
