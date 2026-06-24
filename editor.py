"""lumen3d Editor -- Roblox Studio-style scene editor.

Layout (two SDL2 windows, one process):
  UI window  (lumen.Canvas, 380px wide) -- hierarchy + properties + toolbar
  3D window  (GLWindow + GLRenderer3D)  -- scene viewport

Controls:
  Viewport:
    Right-drag         orbit camera
    Middle-drag        pan camera
    Left-click         select object (ray vs AABB)
    F key              frame selected

  Toolbar (UI window):
    +Cube +Sphere +Cyl +Plane  add primitive
    Del                        delete selected
    Save / Load                scene file
    Play                       run scene in subprocess

  Properties panel:
    name, position, rotation, scale, anchored, gravity, restitution, color
"""
from __future__ import annotations

import math
import subprocess

import lumen
import lumen.gl as gl
from lumen._canvas import Canvas as LumenCanvas

from pugtk._vector import Vector3
from pugtk._camera import Camera
from pugtk._mesh import Mesh
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d._instance import Instance
from lumen3d._world import World
from lumen3d._material import Material
from lumen3d._scene_io import SceneIO
from lumen3d._aabb import world_aabb, AABB

import _gui_sdl

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

UI_W: int = 380
UI_H: int = 780
VP_W: int = 900
VP_H: int = 780

PANEL_BG:    int = 0x1E1E2E
PANEL_BG2:   int = 0x252535
TOOLBAR_BG:  int = 0x181828
BORDER:      int = 0x3A3A5A
SELECTED_BG: int = 0x2A4A7A
HOVER_BG:    int = 0x2A2A4A
TEXT_COL:    int = 0xDDDDEE
TEXT_DIM:    int = 0x8888AA
ACCENT:      int = 0x4A8ADA
BTN_BG:      int = 0x2C3E5A
BTN_HOV:     int = 0x3A5080
BTN_PRESS:   int = 0x1A2A40
RED_BTN:     int = 0x7A2222
RED_HOV:     int = 0xAA3333
GREEN_BTN:   int = 0x226622
GREEN_HOV:   int = 0x338833

TOOLBAR_H:   int = 40
HIER_TOP:    int = TOOLBAR_H + 2
HIER_H:      int = 280
PROP_TOP:    int = HIER_TOP + HIER_H + 4
PROP_H:      int = UI_H - PROP_TOP - 2

ITEM_H:      int = 22
CH_W:        int = 8
CH_H:        int = 8

# ---------------------------------------------------------------------------
# Global mutable state (asmpython globals are module-level assignments)
# ---------------------------------------------------------------------------

_edit_field_id: str = ""
_edit_text: str = ""
_selected_name: str = ""

_cam_yaw:   float = 0.4
_cam_pitch: float = 0.5
_cam_dist:  float = 12.0
_cam_cx:    float = 0.0
_cam_cy:    float = 1.0
_cam_cz:    float = 0.0

_orbit_dragging: int = 0
_pan_dragging:   int = 0

_mx: int = 0
_my: int = 0
_mbtn: int = 0
_mbtn_down: int = 0
_last_key: int = 0
_mouse_rel_dx: int = 0
_mouse_rel_dy: int = 0

_hier_scroll: int = 0
_prim_color_idx: int = 0

# ---------------------------------------------------------------------------
# Scancode tables
# ---------------------------------------------------------------------------

_LOWER: list = [
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "n","o","p","q","r","s","t","u","v","w","x","y","z",
]
_UPPER: list = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
]
_DIGITS: list = ["1","2","3","4","5","6","7","8","9"]


def _scancode_to_char(sc: int) -> str:
    shift: int = _gui_sdl.is_key_down(225)
    if sc >= lumen.KEY_A and sc <= lumen.KEY_Z:
        offset: int = sc - lumen.KEY_A
        if shift != 0:
            return _UPPER[offset]
        return _LOWER[offset]
    if sc >= lumen.KEY_1 and sc <= lumen.KEY_9:
        offset2: int = sc - lumen.KEY_1
        return _DIGITS[offset2]
    if sc == lumen.KEY_0:
        return "0"
    if sc == lumen.KEY_SPACE:
        return " "
    if sc == 46:
        return "."
    if sc == 45:
        return "-"
    return ""

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fstr2(f: float) -> str:
    sign: str = ""
    v: float = f
    if v < 0.0:
        sign = "-"
        v = -v
    whole: int = int(v)
    frac: int = int((v - float(whole)) * 100.0 + 0.5)
    frac_s: str = str(frac)
    if frac < 10:
        frac_s = "0" + frac_s
    return sign + str(whole) + "." + frac_s


def _clamp_f(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _hit_rect(mx: int, my: int, x: int, y: int, w: int, h: int) -> int:
    if mx < x:
        return 0
    if mx > x + w:
        return 0
    if my < y:
        return 0
    if my > y + h:
        return 0
    return 1

# ---------------------------------------------------------------------------
# Flat instance list
# ---------------------------------------------------------------------------

def _collect_flat(inst: Instance, depth: int, out_names: list, out_depths: list, out_insts: list) -> None:
    out_names.append(inst.name)
    out_depths.append(depth)
    out_insts.append(inst)
    ci: int = 0
    while ci < len(inst.children):
        _collect_flat(inst.children[ci], depth + 1, out_names, out_depths, out_insts)
        ci = ci + 1


def _build_flat(world: World, out_names: list, out_depths: list, out_insts: list) -> None:
    ri: int = 0
    while ri < len(world.roots):
        _collect_flat(world.roots[ri], 0, out_names, out_depths, out_insts)
        ri = ri + 1


def _find_by_name(world: World, name: str) -> Instance:
    names: list = []
    depths: list = []
    insts: list = []
    _build_flat(world, names, depths, insts)
    i: int = 0
    while i < len(insts):
        if insts[i].name == name:
            return insts[i]
        i = i + 1
    return None


def _remove_from_world(world: World, inst: Instance) -> None:
    if inst.parent is not None:
        new_ch: list = []
        ci: int = 0
        while ci < len(inst.parent.children):
            if inst.parent.children[ci].name != inst.name:
                new_ch.append(inst.parent.children[ci])
            ci = ci + 1
        inst.parent.children = new_ch
        inst.parent = None
    else:
        new_roots: list = []
        ri: int = 0
        while ri < len(world.roots):
            if world.roots[ri].name != inst.name:
                new_roots.append(world.roots[ri])
            ri = ri + 1
        world.roots = new_roots


def _unique_name(world: World, base: str) -> str:
    names: list = []
    depths: list = []
    insts: list = []
    _build_flat(world, names, depths, insts)
    n: int = 1
    candidate: str = base + str(n)
    found: int = 1
    while found == 1:
        found = 0
        i: int = 0
        while i < len(names):
            if names[i] == candidate:
                found = 1
            i = i + 1
        if found == 1:
            n = n + 1
            candidate = base + str(n)
    return candidate

# ---------------------------------------------------------------------------
# Primitive colours
# ---------------------------------------------------------------------------

_PRIM_COLORS: list = [0x6688CC, 0xCC8844, 0x44AA66, 0xAA6688, 0x88AACC, 0xCCAA44]


def _next_color() -> int:
    global _prim_color_idx
    c: int = _PRIM_COLORS[_prim_color_idx % len(_PRIM_COLORS)]
    _prim_color_idx = _prim_color_idx + 1
    return c

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _update_camera(camera: Camera) -> None:
    cx: float = _cam_cx + _cam_dist * math.cos(_cam_pitch) * math.sin(_cam_yaw)
    cy: float = _cam_cy + _cam_dist * math.sin(_cam_pitch)
    cz: float = _cam_cz + _cam_dist * math.cos(_cam_pitch) * math.cos(_cam_yaw)
    camera.position = Vector3(cx, cy, cz)
    camera.target = Vector3(_cam_cx, _cam_cy, _cam_cz)


def _frame_inst(inst: Instance) -> None:
    global _cam_cx, _cam_cy, _cam_cz, _cam_dist
    _cam_cx = inst.position.x
    _cam_cy = inst.position.y
    _cam_cz = inst.position.z
    _cam_dist = 6.0

# ---------------------------------------------------------------------------
# Ray picking
# ---------------------------------------------------------------------------

def _screen_ray_dir(mx: int, my: int, camera: Camera) -> Vector3:
    ndc_x: float = (float(mx) / float(VP_W)) * 2.0 - 1.0
    ndc_y: float = 1.0 - (float(my) / float(VP_H)) * 2.0
    tan_half: float = math.tan(camera.fov_deg * math.pi / 180.0 * 0.5)
    ray_dx: float = ndc_x * camera.aspect * tan_half
    ray_dy: float = ndc_y * tan_half
    fwd_x: float = camera.target.x - camera.position.x
    fwd_y: float = camera.target.y - camera.position.y
    fwd_z: float = camera.target.z - camera.position.z
    flen: float = math.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + fwd_z * fwd_z)
    if flen < 0.0001:
        flen = 0.0001
    fwd_x = fwd_x / flen
    fwd_y = fwd_y / flen
    fwd_z = fwd_z / flen
    right_x: float = fwd_y * 0.0 - fwd_z * 1.0
    right_y: float = fwd_z * 0.0 - fwd_x * 0.0
    right_z: float = fwd_x * 1.0 - fwd_y * 0.0
    rlen: float = math.sqrt(right_x * right_x + right_y * right_y + right_z * right_z)
    if rlen < 0.0001:
        rlen = 0.0001
    right_x = right_x / rlen
    right_y = right_y / rlen
    right_z = right_z / rlen
    up_x: float = right_y * fwd_z - right_z * fwd_y
    up_y: float = right_z * fwd_x - right_x * fwd_z
    up_z: float = right_x * fwd_y - right_y * fwd_x
    dir_x: float = fwd_x + right_x * ray_dx + up_x * ray_dy
    dir_y: float = fwd_y + right_y * ray_dx + up_y * ray_dy
    dir_z: float = fwd_z + right_z * ray_dx + up_z * ray_dy
    dlen: float = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    if dlen < 0.0001:
        dlen = 0.0001
    return Vector3(dir_x / dlen, dir_y / dlen, dir_z / dlen)


def _ray_aabb_t(ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, ab: AABB) -> float:
    inv_dx: float = 1.0 / dx if dx != 0.0 else 1e18
    inv_dy: float = 1.0 / dy if dy != 0.0 else 1e18
    inv_dz: float = 1.0 / dz if dz != 0.0 else 1e18
    t1x: float = (ab.min_x - ox) * inv_dx
    t2x: float = (ab.max_x - ox) * inv_dx
    t1y: float = (ab.min_y - oy) * inv_dy
    t2y: float = (ab.max_y - oy) * inv_dy
    t1z: float = (ab.min_z - oz) * inv_dz
    t2z: float = (ab.max_z - oz) * inv_dz
    tmin_x: float = t1x if t1x < t2x else t2x
    tmax_x: float = t2x if t1x < t2x else t1x
    tmin_y: float = t1y if t1y < t2y else t2y
    tmax_y: float = t2y if t1y < t2y else t1y
    tmin_z: float = t1z if t1z < t2z else t2z
    tmax_z: float = t2z if t1z < t2z else t1z
    tmin: float = tmin_x
    if tmin_y > tmin:
        tmin = tmin_y
    if tmin_z > tmin:
        tmin = tmin_z
    tmax: float = tmax_x
    if tmax_y < tmax:
        tmax = tmax_y
    if tmax_z < tmax:
        tmax = tmax_z
    if tmax < 0.0:
        return -1.0
    if tmin > tmax:
        return -1.0
    if tmin < 0.0:
        return tmax
    return tmin


def _pick(world: World, camera: Camera, mx: int, my: int) -> str:
    names: list = []
    depths: list = []
    insts: list = []
    _build_flat(world, names, depths, insts)
    d: Vector3 = _screen_ray_dir(mx, my, camera)
    ox: float = camera.position.x
    oy: float = camera.position.y
    oz: float = camera.position.z
    best_t: float = -1.0
    best_name: str = ""
    i: int = 0
    while i < len(insts):
        inst: Instance = insts[i]
        if inst.mesh is not None:
            ab: AABB = world_aabb(inst.mesh, inst.world_matrix())
            t: float = _ray_aabb_t(ox, oy, oz, d.x, d.y, d.z, ab)
            if t > 0.0:
                if best_t < 0.0 or t < best_t:
                    best_t = t
                    best_name = inst.name
        i = i + 1
    return best_name

# ---------------------------------------------------------------------------
# UI drawing
# ---------------------------------------------------------------------------

def _btn(canvas: LumenCanvas, x: int, y: int, w: int, h: int, label: str,
         hovered: int, pressed: int, danger: int) -> None:
    bg: int = BTN_BG
    if danger == 1:
        bg = RED_BTN
        if hovered == 1:
            bg = RED_HOV
    elif hovered == 1:
        bg = BTN_HOV
    if pressed == 1:
        bg = BTN_PRESS
    canvas.color(bg)
    canvas.fill(x, y, w, h)
    canvas.color(BORDER)
    canvas.rect(x, y, w, h)
    tx: int = x + (w - len(label) * CH_W) // 2
    ty: int = y + (h - CH_H) // 2
    canvas.color(TEXT_COL)
    canvas.text(tx, ty, label)


def _text_field(canvas: LumenCanvas, x: int, y: int, w: int, h: int,
                field_id: str, value: str) -> None:
    active: int = 1 if _edit_field_id == field_id else 0
    bg: int = 0x111122 if active == 1 else 0x0A0A18
    canvas.color(bg)
    canvas.fill(x, y, w, h)
    canvas.color(ACCENT if active == 1 else BORDER)
    canvas.rect(x, y, w, h)
    display: str = _edit_text if active == 1 else value
    canvas.color(TEXT_COL)
    canvas.text(x + 3, y + (h - CH_H) // 2, display)


def _checkbox(canvas: LumenCanvas, x: int, y: int, checked: int, label: str) -> None:
    canvas.color(BORDER)
    canvas.rect(x, y, 14, 14)
    if checked == 1:
        canvas.color(ACCENT)
        canvas.fill(x + 3, y + 3, 8, 8)
    canvas.color(TEXT_COL)
    canvas.text(x + 18, y + 3, label)

# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------

_TOOLBAR_BTNS: list = [
    "+Cube", "+Sphere", "+Cyl", "+Plane", "Del", "Save", "Load", "Play",
]
_TB_W: int = 44
_TB_PAD: int = 3


def _draw_toolbar(canvas: LumenCanvas, hovered_btn: int, pressed_btn: int) -> None:
    canvas.color(TOOLBAR_BG)
    canvas.fill(0, 0, UI_W, TOOLBAR_H)
    canvas.color(BORDER)
    canvas.line(0, TOOLBAR_H - 1, UI_W, TOOLBAR_H - 1)
    i: int = 0
    while i < len(_TOOLBAR_BTNS):
        bx: int = _TB_PAD + i * (_TB_W + _TB_PAD)
        by: int = 4
        bh: int = TOOLBAR_H - 8
        label: str = _TOOLBAR_BTNS[i]
        danger: int = 1 if label == "Del" else 0
        _btn(canvas, bx, by, _TB_W, bh, label,
             1 if hovered_btn == i else 0,
             1 if pressed_btn == i else 0,
             danger)
        i = i + 1


def _toolbar_hit(mx: int, my: int) -> int:
    if my < 0 or my > TOOLBAR_H:
        return -1
    i: int = 0
    while i < len(_TOOLBAR_BTNS):
        bx: int = _TB_PAD + i * (_TB_W + _TB_PAD)
        if _hit_rect(mx, my, bx, 4, _TB_W, TOOLBAR_H - 8) == 1:
            return i
        i = i + 1
    return -1

# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

def _draw_hierarchy(canvas: LumenCanvas, world: World, sel_name: str, hover_y: int) -> None:
    canvas.color(PANEL_BG)
    canvas.fill(0, HIER_TOP, UI_W, HIER_H)
    canvas.color(BORDER)
    canvas.rect(0, HIER_TOP, UI_W - 1, HIER_H)
    canvas.color(ACCENT)
    canvas.text(6, HIER_TOP + 4, "HIERARCHY")
    canvas.color(BORDER)
    canvas.line(0, HIER_TOP + 16, UI_W - 1, HIER_TOP + 16)

    names: list = []
    depths: list = []
    insts: list = []
    _build_flat(world, names, depths, insts)

    visible_top: int = HIER_TOP + 18
    max_rows: int = (HIER_H - 20) // ITEM_H
    i: int = _hier_scroll
    row: int = 0
    while i < len(names) and row < max_rows:
        iy: int = visible_top + row * ITEM_H
        indent: int = 6 + depths[i] * 14
        is_sel: int = 1 if names[i] == sel_name else 0
        is_hov: int = 1 if iy <= hover_y and hover_y < iy + ITEM_H else 0
        if is_sel == 1:
            canvas.color(SELECTED_BG)
            canvas.fill(0, iy, UI_W - 2, ITEM_H)
        elif is_hov == 1:
            canvas.color(HOVER_BG)
            canvas.fill(0, iy, UI_W - 2, ITEM_H)
        if len(insts[i].children) > 0:
            canvas.color(TEXT_DIM)
            canvas.text(indent, iy + 7, ">")
        canvas.color(TEXT_COL if is_sel == 1 else TEXT_DIM)
        canvas.text(indent + 10, iy + 7, names[i])
        i = i + 1
        row = row + 1


def _hierarchy_click(world: World, my: int) -> str:
    names: list = []
    depths: list = []
    insts: list = []
    _build_flat(world, names, depths, insts)
    visible_top: int = HIER_TOP + 18
    max_rows: int = (HIER_H - 20) // ITEM_H
    i: int = _hier_scroll
    row: int = 0
    while i < len(names) and row < max_rows:
        iy: int = visible_top + row * ITEM_H
        if my >= iy and my < iy + ITEM_H:
            return names[i]
        i = i + 1
        row = row + 1
    return ""

# ---------------------------------------------------------------------------
# Properties (no nested functions -- all inlined)
# ---------------------------------------------------------------------------

_PROP_PAD: int = 6
_PROP_LW:  int = 56
_PROP_FH:  int = 16


def _inst_color(inst: Instance) -> int:
    if len(inst.colors) > 0:
        return inst.colors[0]
    return 0xC0C0C0


def _inst_field_value(inst: Instance, fid: str) -> str:
    if fid == "name":
        return inst.name
    if fid == "px":
        return _fstr2(inst.position.x)
    if fid == "py":
        return _fstr2(inst.position.y)
    if fid == "pz":
        return _fstr2(inst.position.z)
    if fid == "rx":
        return _fstr2(inst.rotation.x)
    if fid == "ry":
        return _fstr2(inst.rotation.y)
    if fid == "rz":
        return _fstr2(inst.rotation.z)
    if fid == "sx":
        return _fstr2(inst.scale.x)
    if fid == "sy":
        return _fstr2(inst.scale.y)
    if fid == "sz":
        return _fstr2(inst.scale.z)
    if fid == "rest":
        return _fstr2(inst.restitution)
    if fid == "color":
        return hex(_inst_color(inst))
    return ""


def _apply_field(inst: Instance, fid: str, val: str) -> None:
    if fid == "name":
        inst.name = val
        return
    if fid == "color":
        raw: str = val
        if len(raw) > 2 and (raw[0:2] == "0x" or raw[0:2] == "0X"):
            raw = raw[2:]
        col: int = int(raw, 16)
        mat: Material = Material("edited")
        mat.color = col
        inst.set_material(mat)
        return
    v: float = float(val)
    if fid == "px":
        inst.position = Vector3(v, inst.position.y, inst.position.z)
    elif fid == "py":
        inst.position = Vector3(inst.position.x, v, inst.position.z)
    elif fid == "pz":
        inst.position = Vector3(inst.position.x, inst.position.y, v)
    elif fid == "rx":
        inst.rotation = Vector3(v, inst.rotation.y, inst.rotation.z)
    elif fid == "ry":
        inst.rotation = Vector3(inst.rotation.x, v, inst.rotation.z)
    elif fid == "rz":
        inst.rotation = Vector3(inst.rotation.x, inst.rotation.y, v)
    elif fid == "sx":
        inst.scale = Vector3(v, inst.scale.y, inst.scale.z)
    elif fid == "sy":
        inst.scale = Vector3(inst.scale.x, v, inst.scale.z)
    elif fid == "sz":
        inst.scale = Vector3(inst.scale.x, inst.scale.y, v)
    elif fid == "rest":
        inst.restitution = v


def _draw_properties(canvas: LumenCanvas, inst: Instance) -> None:
    canvas.color(PANEL_BG2)
    canvas.fill(0, PROP_TOP, UI_W, PROP_H)
    canvas.color(BORDER)
    canvas.rect(0, PROP_TOP, UI_W - 1, PROP_H)
    canvas.color(ACCENT)
    canvas.text(6, PROP_TOP + 4, "PROPERTIES")
    canvas.color(BORDER)
    canvas.line(0, PROP_TOP + 16, UI_W - 1, PROP_TOP + 16)

    if inst is None:
        canvas.color(TEXT_DIM)
        canvas.text(10, PROP_TOP + 30, "No selection")
        return

    pad: int = _PROP_PAD
    lw: int = _PROP_LW
    fh: int = _PROP_FH
    fw: int = UI_W - lw - pad * 3
    fw3: int = (UI_W - lw - pad * 3 - 4) // 3

    y: int = PROP_TOP + 22

    canvas.color(TEXT_DIM)
    canvas.text(pad, y + 4, "name")
    _text_field(canvas, pad + lw, y, fw, fh, "name", inst.name)
    y = y + fh + 5

    canvas.color(TEXT_DIM)
    canvas.text(pad, y, "-- TRANSFORM --")
    y = y + 11

    canvas.text(pad, y + 4, "pos")
    canvas.text(pad + lw, y + 4, "X")
    _text_field(canvas, pad + lw + 10, y, fw3 - 10, fh, "px", _inst_field_value(inst, "px"))
    canvas.text(pad + lw + fw3 + 4, y + 4, "Y")
    _text_field(canvas, pad + lw + fw3 + 14, y, fw3 - 10, fh, "py", _inst_field_value(inst, "py"))
    canvas.text(pad + lw + fw3 * 2 + 8, y + 4, "Z")
    _text_field(canvas, pad + lw + fw3 * 2 + 18, y, fw3 - 16, fh, "pz", _inst_field_value(inst, "pz"))
    y = y + fh + 3

    canvas.text(pad, y + 4, "rot")
    canvas.text(pad + lw, y + 4, "X")
    _text_field(canvas, pad + lw + 10, y, fw3 - 10, fh, "rx", _inst_field_value(inst, "rx"))
    canvas.text(pad + lw + fw3 + 4, y + 4, "Y")
    _text_field(canvas, pad + lw + fw3 + 14, y, fw3 - 10, fh, "ry", _inst_field_value(inst, "ry"))
    canvas.text(pad + lw + fw3 * 2 + 8, y + 4, "Z")
    _text_field(canvas, pad + lw + fw3 * 2 + 18, y, fw3 - 16, fh, "rz", _inst_field_value(inst, "rz"))
    y = y + fh + 3

    canvas.text(pad, y + 4, "scale")
    canvas.text(pad + lw, y + 4, "X")
    _text_field(canvas, pad + lw + 10, y, fw3 - 10, fh, "sx", _inst_field_value(inst, "sx"))
    canvas.text(pad + lw + fw3 + 4, y + 4, "Y")
    _text_field(canvas, pad + lw + fw3 + 14, y, fw3 - 10, fh, "sy", _inst_field_value(inst, "sy"))
    canvas.text(pad + lw + fw3 * 2 + 8, y + 4, "Z")
    _text_field(canvas, pad + lw + fw3 * 2 + 18, y, fw3 - 16, fh, "sz", _inst_field_value(inst, "sz"))
    y = y + fh + 7

    canvas.color(TEXT_DIM)
    canvas.text(pad, y, "-- PHYSICS --")
    y = y + 11

    _checkbox(canvas, pad, y, inst.anchored, "Anchored")
    _checkbox(canvas, pad + 110, y, inst.gravity_enabled, "Gravity")
    y = y + 20

    canvas.color(TEXT_DIM)
    canvas.text(pad, y + 4, "bounce")
    _text_field(canvas, pad + lw, y, 64, fh, "rest", _inst_field_value(inst, "rest"))
    y = y + fh + 7

    canvas.color(TEXT_DIM)
    canvas.text(pad, y, "-- APPEARANCE --")
    y = y + 11

    col_val: int = _inst_color(inst)
    canvas.text(pad, y + 4, "color")
    _text_field(canvas, pad + lw, y, 96, fh, "color", hex(col_val))
    canvas.color(col_val)
    canvas.fill(pad + lw + 100, y, 22, fh)
    canvas.color(BORDER)
    canvas.rect(pad + lw + 100, y, 22, fh)


def _prop_field_at(my: int, mx: int) -> str:
    if my < PROP_TOP or my > PROP_TOP + PROP_H:
        return ""

    pad: int = _PROP_PAD
    lw: int = _PROP_LW
    fh: int = _PROP_FH
    fw: int = UI_W - lw - pad * 3
    fw3: int = (UI_W - lw - pad * 3 - 4) // 3

    y: int = PROP_TOP + 22

    name_y: int = y
    y = y + fh + 5
    if my >= name_y and my < name_y + fh:
        return "name"

    y = y + 11

    pos_y: int = y
    y = y + fh + 3
    if my >= pos_y and my < pos_y + fh:
        if mx >= pad + lw + 10 and mx < pad + lw + fw3:
            return "px"
        if mx >= pad + lw + fw3 + 14 and mx < pad + lw + fw3 * 2 + 4:
            return "py"
        if mx >= pad + lw + fw3 * 2 + 18:
            return "pz"
        return ""

    rot_y: int = y
    y = y + fh + 3
    if my >= rot_y and my < rot_y + fh:
        if mx >= pad + lw + 10 and mx < pad + lw + fw3:
            return "rx"
        if mx >= pad + lw + fw3 + 14 and mx < pad + lw + fw3 * 2 + 4:
            return "ry"
        if mx >= pad + lw + fw3 * 2 + 18:
            return "rz"
        return ""

    scl_y: int = y
    y = y + fh + 7 + 11 + 20

    if my >= scl_y and my < scl_y + fh:
        if mx >= pad + lw + 10 and mx < pad + lw + fw3:
            return "sx"
        if mx >= pad + lw + fw3 + 14 and mx < pad + lw + fw3 * 2 + 4:
            return "sy"
        if mx >= pad + lw + fw3 * 2 + 18:
            return "sz"
        return ""

    rest_y: int = y
    y = y + fh + 7 + 11
    if my >= rest_y and my < rest_y + fh:
        return "rest"

    col_y: int = y
    if my >= col_y and my < col_y + fh:
        return "color"

    return ""


def _prop_checkbox_at(inst: Instance, mx: int, my: int) -> str:
    if inst is None:
        return ""
    pad: int = _PROP_PAD
    lw: int = _PROP_LW
    fh: int = _PROP_FH
    y: int = PROP_TOP + 22 + fh + 5 + 11 + (fh + 3) * 3 + 7 + 11
    if my >= y and my < y + 20:
        if _hit_rect(mx, my, pad, y, 100, 14) == 1:
            return "anchored"
        if _hit_rect(mx, my, pad + 110, y, 100, 14) == 1:
            return "gravity"
    return ""

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

_MESH_TYPES: dict = {}


def _add_primitive(world: World, prim: str) -> Instance:
    name: str = _unique_name(world, prim)
    mat: Material = Material(name)
    mat.color = _next_color()
    mesh: Mesh = Mesh.cube(1.0)
    if prim == "Sphere":
        mesh = Mesh.sphere(0.5, 10, 14)
    elif prim == "Cylinder":
        mesh = Mesh.cylinder(0.5, 1.0, 12)
    elif prim == "Plane":
        mesh = Mesh.plane(4.0)
    inst: Instance = Instance(name, mesh, [])
    inst.set_material(mat)
    world.add(inst)
    _MESH_TYPES[name] = prim
    return inst

# ---------------------------------------------------------------------------
# Scene IO
# ---------------------------------------------------------------------------

_SCENE_PATH: str = "editor_scene.scene"


def _save_scene(world: World) -> None:
    sio: SceneIO = SceneIO()
    sio.save(world, _SCENE_PATH)
    print("Scene saved: " + _SCENE_PATH)


def _load_scene(world: World) -> None:
    meshes: dict = {}
    names: list = []
    depths: list = []
    insts: list = []
    _build_flat(world, names, depths, insts)
    i: int = 0
    while i < len(names):
        t: str = _MESH_TYPES.get(names[i])
        if t is None:
            t = "Cube"
        mesh: Mesh = Mesh.cube(1.0)
        if t == "Sphere":
            mesh = Mesh.sphere(0.5, 10, 14)
        elif t == "Cylinder":
            mesh = Mesh.cylinder(0.5, 1.0, 12)
        elif t == "Plane":
            mesh = Mesh.plane(4.0)
        meshes[names[i]] = mesh
        i = i + 1
    world.roots = []
    sio: SceneIO = SceneIO()
    sio.load(world, _SCENE_PATH, meshes)
    new_names: list = []
    new_depths: list = []
    new_insts: list = []
    _build_flat(world, new_names, new_depths, new_insts)
    ni: int = 0
    while ni < len(new_insts):
        col: int = _inst_color(new_insts[ni])
        if col == 0xC0C0C0:
            mat2: Material = Material(new_names[ni])
            mat2.color = _next_color()
            new_insts[ni].set_material(mat2)
        ni = ni + 1
    print("Scene loaded: " + str(len(new_insts)) + " instances")

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

def _draw_status(canvas: LumenCanvas, sel_name: str) -> None:
    y: int = UI_H - 14
    canvas.color(TOOLBAR_BG)
    canvas.fill(0, y, UI_W, 14)
    canvas.color(TEXT_DIM)
    msg: str = "RMB=orbit  MMB=pan  F=frame  Del=delete"
    if sel_name != "":
        msg = "Selected: " + sel_name + "  |  " + msg
    canvas.text(4, y + 3, msg)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _selected_name, _edit_field_id, _edit_text
    global _cam_yaw, _cam_pitch, _cam_dist, _cam_cx, _cam_cy, _cam_cz
    global _orbit_dragging, _pan_dragging
    global _mx, _my, _mbtn, _mbtn_down, _last_key, _mouse_rel_dx, _mouse_rel_dy

    canvas: LumenCanvas = lumen.Canvas("lumen3d Editor", UI_W, UI_H)
    vp_win: GLWindow = GLWindow("lumen3d Viewport", VP_W, VP_H)
    camera: Camera = Camera(
        Vector3(0.0, 4.0, 12.0),
        Vector3(0.0, 0.0, 0.0),
        Vector3(0.0, 1.0, 0.0),
        60.0,
        float(VP_W) / float(VP_H),
        0.1,
        500.0,
    )
    renderer: GLRenderer3D = GLRenderer3D(vp_win, camera)
    renderer.light_dir = Vector3(0.5, 1.0, 0.7)
    renderer.ambient = 0.25

    world: World = World(renderer)
    world.collision_enabled = 0

    floor_mat: Material = Material("Floor")
    floor_mat.color = 0x444455
    floor_inst: Instance = Instance("Floor", Mesh.plane(20.0), [])
    floor_inst.set_material(floor_mat)
    floor_inst.anchored = 1
    world.add(floor_inst)
    _MESH_TYPES["Floor"] = "Plane"

    running: int = 1
    hovered_btn: int = -1
    pressed_btn: int = -1

    while running == 1:
        _mbtn_down = 0
        _mouse_rel_dx = 0
        _mouse_rel_dy = 0

        ev: int = _gui_sdl.poll_event()
        while ev != 0:
            if ev == 0x100:
                running = 0
            elif ev == 0x300:
                _last_key = _gui_sdl.key_scancode()
            elif ev == 0x400:
                new_mx: int = _gui_sdl.mouse_x()
                new_my: int = _gui_sdl.mouse_y()
                _mouse_rel_dx = new_mx - _mx
                _mouse_rel_dy = new_my - _my
                _mx = new_mx
                _my = new_my
            elif ev == 0x401:
                _mbtn = _gui_sdl.mouse_button()
                _mbtn_down = _mbtn
            elif ev == 0x402:
                _mbtn = 0
            ev = _gui_sdl.poll_event()

        # -- orbit --
        if _mbtn == 3:
            if _orbit_dragging == 0:
                _orbit_dragging = 1
            else:
                _cam_yaw = _cam_yaw + float(_mouse_rel_dx) * 0.007
                _cam_pitch = _cam_pitch - float(_mouse_rel_dy) * 0.007
                _cam_pitch = _clamp_f(_cam_pitch, -1.4, 1.4)
        else:
            _orbit_dragging = 0

        # -- pan --
        if _mbtn == 2:
            if _pan_dragging == 0:
                _pan_dragging = 1
            else:
                fwd_x: float = camera.target.x - camera.position.x
                fwd_z: float = camera.target.z - camera.position.z
                flen: float = math.sqrt(fwd_x * fwd_x + fwd_z * fwd_z)
                if flen > 0.0001:
                    fwd_x = fwd_x / flen
                    fwd_z = fwd_z / flen
                pan_speed: float = _cam_dist * 0.002
                _cam_cx = _cam_cx - fwd_z * float(_mouse_rel_dx) * pan_speed
                _cam_cz = _cam_cz + fwd_x * float(_mouse_rel_dx) * pan_speed
                _cam_cy = _cam_cy + float(_mouse_rel_dy) * pan_speed
        else:
            _pan_dragging = 0

        _update_camera(camera)

        # -- left click in viewport: pick object --
        if _mbtn_down == 1 and _my < VP_H and _mx < VP_W and _mx >= 0:
            hit: str = _pick(world, camera, _mx, _my)
            _selected_name = hit
            _edit_field_id = ""

        ui_mx: int = _mx
        ui_my: int = _my
        hovered_btn = _toolbar_hit(ui_mx, ui_my)

        # -- toolbar click --
        if _mbtn_down == 1:
            tb: int = _toolbar_hit(ui_mx, ui_my)
            if tb >= 0:
                pressed_btn = tb
                lbl: str = _TOOLBAR_BTNS[tb]
                if lbl == "+Cube":
                    ni: Instance = _add_primitive(world, "Cube")
                    _selected_name = ni.name
                elif lbl == "+Sphere":
                    ni2: Instance = _add_primitive(world, "Sphere")
                    _selected_name = ni2.name
                elif lbl == "+Cyl":
                    ni3: Instance = _add_primitive(world, "Cylinder")
                    _selected_name = ni3.name
                elif lbl == "+Plane":
                    ni4: Instance = _add_primitive(world, "Plane")
                    _selected_name = ni4.name
                elif lbl == "Del":
                    if _selected_name != "":
                        di: Instance = _find_by_name(world, _selected_name)
                        if di is not None:
                            _remove_from_world(world, di)
                        _selected_name = ""
                        _edit_field_id = ""
                elif lbl == "Save":
                    _save_scene(world)
                elif lbl == "Load":
                    _load_scene(world)
                elif lbl == "Play":
                    _save_scene(world)
                    subprocess.Popen(["python", "-m", "asmpython", "gl_play_scene.py"])
            else:
                pressed_btn = -1

        # -- hierarchy click --
        if _mbtn_down == 1 and _hit_rect(ui_mx, ui_my, 0, HIER_TOP, UI_W, HIER_H) == 1:
            clicked: str = _hierarchy_click(world, ui_my)
            if clicked != "":
                _selected_name = clicked
                _edit_field_id = ""

        sel_inst: Instance = _find_by_name(world, _selected_name)

        # -- properties click --
        if _mbtn_down == 1 and sel_inst is not None:
            chk: str = _prop_checkbox_at(sel_inst, ui_mx, ui_my)
            if chk == "anchored":
                sel_inst.anchored = 1 if sel_inst.anchored == 0 else 0
            elif chk == "gravity":
                sel_inst.gravity_enabled = 1 if sel_inst.gravity_enabled == 0 else 0
            elif _hit_rect(ui_mx, ui_my, 0, PROP_TOP, UI_W, PROP_H) == 1:
                fid: str = _prop_field_at(ui_my, ui_mx)
                if fid != "":
                    if _edit_field_id != fid:
                        if _edit_field_id != "":
                            _apply_field(sel_inst, _edit_field_id, _edit_text)
                        _edit_field_id = fid
                        _edit_text = _inst_field_value(sel_inst, fid)
                else:
                    if _edit_field_id != "":
                        _apply_field(sel_inst, _edit_field_id, _edit_text)
                    _edit_field_id = ""

        # -- keyboard --
        if _last_key != 0:
            k: int = _last_key
            _last_key = 0
            if _edit_field_id != "":
                if k == lumen.KEY_BACKSPACE:
                    if len(_edit_text) > 0:
                        _edit_text = _edit_text[:len(_edit_text) - 1]
                elif k == lumen.KEY_RETURN:
                    if sel_inst is not None:
                        _apply_field(sel_inst, _edit_field_id, _edit_text)
                    _edit_field_id = ""
                elif k == lumen.KEY_ESCAPE:
                    _edit_field_id = ""
                else:
                    ch: str = _scancode_to_char(k)
                    if ch != "":
                        _edit_text = _edit_text + ch
            else:
                if k == lumen.KEY_DELETE:
                    if _selected_name != "":
                        di2: Instance = _find_by_name(world, _selected_name)
                        if di2 is not None:
                            _remove_from_world(world, di2)
                        _selected_name = ""
                elif k == lumen.KEY_F:
                    if sel_inst is not None:
                        _frame_inst(sel_inst)

        # -- draw UI --
        canvas.color(PANEL_BG)
        canvas.fill(0, 0, UI_W, UI_H)

        _draw_toolbar(canvas, hovered_btn, pressed_btn)
        pressed_btn = -1

        hier_hover_y: int = ui_my if _hit_rect(ui_mx, ui_my, 0, HIER_TOP, UI_W, HIER_H) == 1 else -1
        _draw_hierarchy(canvas, world, _selected_name, hier_hover_y)
        _draw_properties(canvas, sel_inst)
        _draw_status(canvas, _selected_name)

        canvas.present()

        # -- draw 3D --
        world.render()
        gl.swap_window(vp_win.win)

    canvas.close()
    vp_win.close()


main()
