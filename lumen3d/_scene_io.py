"""SceneIO -- save and load a World's Instance tree to/from a simple
text format.

Format (one line per field, one instance per block):

    instance Player
    position 0.0 4.0 0.0
    rotation 0.0 0.0 0.0
    scale 1.0 1.0 1.0
    gravity_enabled 1
    anchored 0
    parent none
    color_count 12
    end

Colors are NOT saved (too verbose, rely on mesh name for asset re-bind).
Mesh and texture are NOT saved (re-referenced by instance name via a
user-supplied meshes dict on load).

Usage:
    sio = SceneIO()
    sio.save(world, "level1.scene")

    meshes = {"Player": player_mesh, "Floor": floor_mesh}
    sio.load(world2, "level1.scene", meshes)
"""
from __future__ import annotations

import io as _io

from pugtk._vector import Vector3
from pugtk._mesh import Mesh

from ._instance import Instance
from ._world import World


def _fstr(f: float) -> str:
    return str(f)


class SceneIO:

    def _write_inst(self, inst: Instance, f: _io.FileIO, parent_name: str) -> None:
        f.write("instance " + inst.name + "\n")
        f.write("position " + _fstr(inst.position.x) + " " + _fstr(inst.position.y) + " " + _fstr(inst.position.z) + "\n")
        f.write("rotation " + _fstr(inst.rotation.x) + " " + _fstr(inst.rotation.y) + " " + _fstr(inst.rotation.z) + "\n")
        f.write("scale " + _fstr(inst.scale.x) + " " + _fstr(inst.scale.y) + " " + _fstr(inst.scale.z) + "\n")
        f.write("gravity " + str(inst.gravity_enabled) + "\n")
        f.write("anchored " + str(inst.anchored) + "\n")
        f.write("restitution " + _fstr(inst.restitution) + "\n")
        f.write("parent " + parent_name + "\n")
        f.write("end\n")
        ci: int = 0
        while ci < len(inst.children):
            child: Instance = inst.children[ci]
            self._write_inst(child, f, inst.name)
            ci = ci + 1

    def save(self, world: World, path: str) -> None:
        """Write the entire Instance tree to a scene file."""
        f = _io.FileIO(path, "w")
        ri: int = 0
        while ri < len(world.roots):
            root: Instance = world.roots[ri]
            self._write_inst(root, f, "none")
            ri = ri + 1
        f.close()

    def load(self, world: World, path: str, meshes: dict) -> None:
        """Read instances from a scene file and add them to world.

        meshes: dict[str, Mesh] mapping Instance.name to Mesh.  Names
        not in the dict get mesh=None."""
        f = _io.FileIO(path, "r")
        content: str = f.read()
        f.close()

        all_inst: dict = {}
        all_parents: dict = {}

        cur_name: str = ""
        cur_px: float = 0.0
        cur_py: float = 0.0
        cur_pz: float = 0.0
        cur_rx: float = 0.0
        cur_ry: float = 0.0
        cur_rz: float = 0.0
        cur_sx: float = 1.0
        cur_sy: float = 1.0
        cur_sz: float = 1.0
        cur_grav: int = 0
        cur_anch: int = 0
        cur_parent: str = "none"
        in_inst: int = 0

        lines: list[str] = content.split("\n")
        li: int = 0
        while li < len(lines):
            line: str = lines[li]
            if len(line) == 0:
                li = li + 1
                continue
            parts: list[str] = line.split(" ")
            cmd: str = parts[0]
            if cmd == "instance":
                cur_name = parts[1]
                cur_px = 0.0
                cur_py = 0.0
                cur_pz = 0.0
                cur_rx = 0.0
                cur_ry = 0.0
                cur_rz = 0.0
                cur_sx = 1.0
                cur_sy = 1.0
                cur_sz = 1.0
                cur_grav = 0
                cur_anch = 0
                cur_rest: float = 0.0
                cur_parent = "none"
                in_inst = 1
            elif cmd == "position" and in_inst == 1:
                cur_px = float(parts[1])
                cur_py = float(parts[2])
                cur_pz = float(parts[3])
            elif cmd == "rotation" and in_inst == 1:
                cur_rx = float(parts[1])
                cur_ry = float(parts[2])
                cur_rz = float(parts[3])
            elif cmd == "scale" and in_inst == 1:
                cur_sx = float(parts[1])
                cur_sy = float(parts[2])
                cur_sz = float(parts[3])
            elif cmd == "gravity" and in_inst == 1:
                cur_grav = int(parts[1])
            elif cmd == "anchored" and in_inst == 1:
                cur_anch = int(parts[1])
            elif cmd == "restitution" and in_inst == 1:
                cur_rest = float(parts[1])
            elif cmd == "parent" and in_inst == 1:
                cur_parent = parts[1]
            elif cmd == "end" and in_inst == 1:
                loaded_mesh: Mesh = meshes.get(cur_name)
                colors: list[int] = []
                if loaded_mesh is not None:
                    default_col: int = 0xC0C0C0
                    ti: int = 0
                    while ti < len(loaded_mesh.triangles):
                        colors.append(default_col)
                        ti = ti + 1
                inst: Instance = Instance(cur_name, loaded_mesh, colors)
                inst._position = Vector3(cur_px, cur_py, cur_pz)
                inst._rotation = Vector3(cur_rx, cur_ry, cur_rz)
                inst._scale = Vector3(cur_sx, cur_sy, cur_sz)
                inst._model_dirty = 1
                inst.gravity_enabled = cur_grav
                inst.anchored = cur_anch
                inst.restitution = cur_rest
                all_inst[cur_name] = inst
                all_parents[cur_name] = cur_parent
                in_inst = 0
            li = li + 1

        keys: list[str] = all_inst.keys()
        ki: int = 0
        while ki < len(keys):
            n: str = keys[ki]
            it: Instance = all_inst[n]
            pname: str = all_parents[n]
            if pname == "none":
                world.add(it)
            else:
                pit: Instance = all_inst.get(pname)
                if pit is not None:
                    pit.add_child(it)
                else:
                    world.add(it)
            ki = ki + 1
