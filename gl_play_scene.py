"""Play-scene runner launched by the editor's Play button.

Loads editor_scene.scene and runs it with physics + collision enabled.
All instances get cube meshes on load (editor saves mesh type separately
in a future version -- for now all loaded instances are cubes).

Controls: Escape to quit.
"""
from pugtk import Vector3, Mesh, Camera
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import Instance, World, GameLoop
from lumen3d._material import Material
from lumen3d._scene_io import SceneIO
from lumen3d._input import KEY_ESCAPE

window = GLWindow("lumen3d - Play", 1024, 720)
camera = Camera(
    Vector3(0.0, 5.0, 14.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    1024.0 / 720.0,
    0.1,
    500.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.5, 1.0, 0.7)
renderer.ambient = 0.25

world = World(renderer)
loop = GameLoop(window, world)

cube_mesh = Mesh.cube(1.0)
plane_mesh = Mesh.plane(20.0)
sphere_mesh = Mesh.sphere(0.5, 10, 14)
cyl_mesh = Mesh.cylinder(0.5, 1.0, 12)

meshes = {}

sio = SceneIO()
sio.load(world, "editor_scene.scene", meshes)

mat_default = Material("Default")
mat_default.color = 0xC0C0C0

names: list = []
depths: list = []
insts: list = []


def _collect(inst, depth: int) -> None:
    names.append(inst.name)
    depths.append(depth)
    insts.append(inst)
    ci: int = 0
    while ci < len(inst.children):
        _collect(inst.children[ci], depth + 1)
        ci = ci + 1


ri: int = 0
while ri < len(world.roots):
    _collect(world.roots[ri], 0)
    ri = ri + 1

fi: int = 0
while fi < len(insts):
    if insts[fi].mesh is None:
        insts[fi].mesh = cube_mesh
        insts[fi].colors = mat_default.make_colors(cube_mesh)
    else:
        insts[fi].set_material(mat_default)
    fi = fi + 1


def on_update(frame: int) -> None:
    if loop.input.is_key_down(KEY_ESCAPE) == 1:
        window.close()


loop.updated.connect(on_update)
loop.run()
