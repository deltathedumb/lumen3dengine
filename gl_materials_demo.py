"""Materials demo -- shows Material and MaterialLibrary in action.

A library of named materials is created, then applied to several
cubes in the scene.  Press R/G/B to change the player cube's material
at runtime, demonstrating live material switching.

Controls:
  R        -- switch to Red material
  G        -- switch to Green material
  B        -- switch to Blue (Metal) material
  Escape   -- quit
"""
from pugtk import Vector3, Mesh, Camera
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import Instance, World, GameLoop, Material, MaterialLibrary
from lumen3d._input import KEY_ESCAPE, KEY_R, KEY_G, KEY_B

window = GLWindow("lumen3dengine materials demo", 800, 600)
camera = Camera(
    Vector3(0.0, 3.0, 10.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    800.0 / 600.0,
    0.1,
    200.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.6, 1.0, 0.5)
renderer.ambient = 0.25

world = World(renderer)
world.collision_enabled = 0
loop = GameLoop(window, world)

lib = MaterialLibrary()

mat_red = Material("Red")
mat_red.color = 0xCC3322

mat_green = Material("Green")
mat_green.color = 0x33AA44

mat_blue = Material("Blue")
mat_blue.color = 0x2244CC
mat_blue.metallic = 1.0
mat_blue.roughness = 0.1

mat_grey = Material("Grey")
mat_grey.color = 0x888899

mat_gold = Material("Gold")
mat_gold.color = 0xCCA040
mat_gold.metallic = 0.9

lib.add(mat_red)
lib.add(mat_green)
lib.add(mat_blue)
lib.add(mat_grey)
lib.add(mat_gold)

cube_mesh = Mesh.cube(1.0)

player = Instance("Player", cube_mesh, [])
player.set_material(mat_red)
player.position = Vector3(0.0, 0.5, 0.0)
world.add(player)

left_cube = Instance("Left", cube_mesh, [])
left_cube.set_material(mat_gold)
left_cube.position = Vector3(-3.0, 0.0, 0.0)
world.add(left_cube)

right_cube = Instance("Right", cube_mesh, [])
right_cube.set_material(mat_green)
right_cube.position = Vector3(3.0, 0.0, 0.0)
world.add(right_cube)

back_cube = Instance("Back", cube_mesh, [])
back_cube.set_material(mat_blue)
back_cube.position = Vector3(0.0, 0.0, -3.0)
world.add(back_cube)

floor_mesh = Mesh.cube(1.0)
floor_inst = Instance("Floor", floor_mesh, [])
floor_inst.set_material(mat_grey)
floor_inst.position = Vector3(0.0, -1.0, 0.0)
floor_inst.scale = Vector3(10.0, 0.25, 10.0)
world.add(floor_inst)

print("Materials demo: press R/G/B to change player cube color")
print("Available materials:")
names: list[str] = lib.names()
ni: int = 0
while ni < len(names):
    print(names[ni])
    ni = ni + 1


def on_update(frame: int) -> None:
    inp = loop.input
    angle: float = float(frame) * 0.01
    player.rotation = Vector3(angle * 0.5, angle, 0.0)

    if inp.is_key_down(KEY_R):
        r_mat: Material = lib.get("Red")
        if r_mat is not None:
            player.set_material(r_mat)
    elif inp.is_key_down(KEY_G):
        g_mat: Material = lib.get("Green")
        if g_mat is not None:
            player.set_material(g_mat)
    elif inp.is_key_down(KEY_B):
        b_mat: Material = lib.get("Blue")
        if b_mat is not None:
            player.set_material(b_mat)

    if inp.is_key_down(KEY_ESCAPE):
        window.close()


loop.updated.connect(on_update)
loop.run()
