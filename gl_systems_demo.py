"""Systems demo -- tests Tween, HUD, Particles, StateMachine, Tags, physics helpers.

Scene:
  - Floor + 4 walls (anchored, tagged "Wall")
  - Player sphere (WASD + Space, third-person camera)
  - Floating platform that tweens up/down on a loop
  - Coin cubes (tagged "Coin") that pop and spawn particles on touch
  - HUD: HP bar, coin counter, FPS, debug log
  - StateMachine: Playing -> Paused (P key) -> Playing

Controls:
  WASD       -- move
  Space      -- jump
  P          -- pause/resume
  Escape     -- quit
"""
from __future__ import annotations

import math

from pugtk import Vector3, Mesh, Camera
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D
from pugtk._light import PointLight

from lumen3d import (
    Instance, World, GameLoop, Material,
    ThirdPersonCamera,
    Tween, TweenManager, EASE_OUT_BOUNCE, EASE_IN_OUT_SINE,
    GameState, StateMachine,
    apply_impulse, is_grounded,
    HUD, ParticleEmitter,
    DebugStats, DebugLog,
    tag, get_tagged, has_tag, untag,
)
from lumen3d._input import (
    KEY_ESCAPE, KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE, KEY_P,
)

SPEED: float = 5.0
JUMP_VEL: float = 7.0
COIN_COUNT: int = 5
HP_MAX: int = 100

# ---- setup ----

window = GLWindow("lumen3d Systems Demo", 1024, 720)
camera = Camera(
    Vector3(0.0, 5.0, 12.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    65.0,
    1024.0 / 720.0,
    0.1,
    300.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.5, 1.0, 0.6)
renderer.ambient = 0.2

world = World(renderer)
loop = GameLoop(window, world)

hud = HUD(1024, 720, "lumen3d HUD")
stats = DebugStats(30)
dbg = DebugLog(12)
tweens = TweenManager()

# ---- materials ----

def mat(name: str, col: int) -> Material:
    m: Material = Material(name)
    m.color = col
    return m

mat_floor  = mat("Floor",  0x334433)
mat_wall   = mat("Wall",   0x334455)
mat_player = mat("Player", 0x44AADD)
mat_plat   = mat("Plat",   0xAA8844)
mat_coin   = mat("Coin",   0xFFCC00)
mat_dead   = mat("Dead",   0x444444)

# ---- floor ----

floor = Instance("Floor", Mesh.plane(30.0), [])
floor.set_material(mat_floor)
floor.anchored = 1
world.add(floor)
tag(floor, "Ground")

# ---- walls ----

wall_mesh = Mesh.cube(1.0)

def make_wall(name: str, px: float, py: float, pz: float, sx: float, sy: float, sz: float) -> Instance:
    w: Instance = Instance(name, wall_mesh, [])
    w.set_material(mat_wall)
    w.anchored = 1
    w.position = Vector3(px, py, pz)
    w.scale = Vector3(sx, sy, sz)
    world.add(w)
    tag(w, "Wall")
    return w

make_wall("WallN",  0.0,  2.0, -15.0, 30.0, 4.0, 0.5)
make_wall("WallS",  0.0,  2.0,  15.0, 30.0, 4.0, 0.5)
make_wall("WallW", -15.0, 2.0,   0.0,  0.5, 4.0, 30.0)
make_wall("WallE",  15.0, 2.0,   0.0,  0.5, 4.0, 30.0)

# ---- player ----

player = Instance("Player", Mesh.sphere(0.5, 10, 14), [])
player.set_material(mat_player)
player.gravity_enabled = 1
player.position = Vector3(0.0, 1.0, 0.0)
player.restitution = 0.0
world.add(player)
tag(player, "Player")

cam_ctrl = ThirdPersonCamera(camera, player)
cam_ctrl.distance = 8.0
cam_ctrl.height = 1.0

# ---- floating platform (tweened) ----

plat = Instance("Platform", Mesh.cube(1.0), [])
plat.set_material(mat_plat)
plat.anchored = 1
plat.position = Vector3(5.0, 1.0, 0.0)
plat.scale = Vector3(3.0, 0.4, 3.0)
world.add(plat)
tag(plat, "Platform")

plat_tween: Tween = tweens.tween(plat, "position_y", 0.5, 4.0, 2.0, EASE_IN_OUT_SINE)
plat_tween.loop = 1
plat_tween.play()

plat_tween2: Tween = tweens.tween(plat, "position_y", 4.0, 0.5, 2.0, EASE_IN_OUT_SINE)
plat_tween2.loop = 0

def _chain_plat(v: int) -> None:
    plat_tween2.play()
plat_tween.finished.connect(_chain_plat)

# ---- coins ----

coin_positions: list = [
    Vector3(-4.0, 1.0,  0.0),
    Vector3( 4.0, 1.0,  4.0),
    Vector3(-3.0, 1.0, -4.0),
    Vector3( 0.0, 1.0,  5.0),
    Vector3( 6.0, 1.0, -3.0),
]
coins: list = []
ci: int = 0
while ci < COIN_COUNT:
    cname: str = "Coin" + str(ci)
    coin: Instance = Instance(cname, Mesh.cube(0.5), [])
    coin.set_material(mat_coin)
    coin.anchored = 1
    coin.position = coin_positions[ci]
    world.add(coin)
    tag(coin, "Coin")
    coins.append(coin)
    ci = ci + 1

# ---- particle emitter for coin collect ----

sparks = ParticleEmitter(world, "Spark", 60)
sparks.emit_rate = 0.0
sparks.lifetime = 0.5
sparks.speed = 4.0
sparks.spread = 1.8
sparks.color_start = 0xFFEE44
sparks.color_end = 0xFF4400
sparks.scale_start = 0.15
sparks.scale_end = 0.0
sparks.gravity = 1

# ---- game state ----

coins_collected: int = 0
hp: int = HP_MAX

class PlayingState(GameState):
    def on_enter(self) -> None:
        dbg.info("Playing")

    def on_update(self, dt: float, frame: int) -> None:
        global coins_collected, hp

        inp = loop.input
        cam_ctrl.update(inp)

        yaw: float = cam_ctrl.yaw
        fwd_x: float = math.sin(yaw)
        fwd_z: float = math.cos(yaw)
        right_x: float = math.cos(yaw)
        right_z: float = -math.sin(yaw)

        mx: float = 0.0
        mz: float = 0.0
        if inp.is_key_down(KEY_W) == 1:
            mx = mx + fwd_x
            mz = mz + fwd_z
        if inp.is_key_down(KEY_S) == 1:
            mx = mx - fwd_x
            mz = mz - fwd_z
        if inp.is_key_down(KEY_A) == 1:
            mx = mx - right_x
            mz = mz - right_z
        if inp.is_key_down(KEY_D) == 1:
            mx = mx + right_x
            mz = mz + right_z

        player.velocity = Vector3(mx * SPEED, player.velocity.y, mz * SPEED)

        if inp.is_key_down(KEY_SPACE) == 1:
            if is_grounded(player, 0.5) == 1:
                apply_impulse(player, Vector3(0.0, JUMP_VEL, 0.0))

        tweens.update(dt)
        sparks.update(dt)

        coi: int = 0
        while coi < len(coins):
            coin2: Instance = coins[coi]
            if coin2.scale.x > 0.01:
                dx: float = player.position.x - coin2.position.x
                dy: float = player.position.y - coin2.position.y
                dz: float = player.position.z - coin2.position.z
                dist: float = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < 1.0:
                    coin2.scale = Vector3(0.0, 0.0, 0.0)
                    coins_collected = coins_collected + 1
                    sparks.position = coin2.position
                    sparks.burst(20)
                    dbg.info("Coin! total=" + str(coins_collected))
                    untag(coin2, "Coin")
            coi = coi + 1

        plat_col: Instance = plat
        pdx: float = player.position.x - plat_col.position.x
        pdz: float = player.position.z - plat_col.position.z
        if pdx < 1.5 and pdx > -1.5 and pdz < 1.5 and pdz > -1.5:
            py: float = player.position.y
            ppy: float = plat_col.position.y + 0.2
            if py < ppy + 0.6 and py > ppy - 0.6:
                player.position = Vector3(player.position.x, ppy + 0.5, player.position.z)
                if player.velocity.y < 0.0:
                    player.velocity = Vector3(player.velocity.x, 0.0, player.velocity.z)

        if player.position.y < -5.0:
            player.position = Vector3(0.0, 2.0, 0.0)
            player.velocity = Vector3(0.0, 0.0, 0.0)
            hp = hp - 10
            dbg.warn("Fell! HP=" + str(hp))

        if inp.is_key_down(KEY_P) == 1:
            self.machine.push(PausedState())

        if inp.is_key_down(KEY_ESCAPE) == 1:
            window.close()

        stats.tick(dt)

        hud.clear(0)
        hud.panel(4, 4, 200, 20, 0x222233, 0x444466)
        hud.bar(8, 8, 192, 12, float(hp) / float(HP_MAX), 0x44DD44, 0x222222)
        hud.text(8, 12, "HP", 0xFFFFFF, 1)
        hud.text(4, 28, "Coins: " + str(coins_collected) + "/" + str(COIN_COUNT), 0xFFCC00, 1)
        stats.draw(hud, 4, 40)
        hud.crosshair(512, 360, 10, 0xFFFFFF)
        dbg.draw(hud, 4, 600)
        hud.present()

    def on_exit(self) -> None:
        dbg.info("Leaving Playing")

    def on_pause(self) -> None:
        dbg.info("Paused")

    def on_resume(self) -> None:
        dbg.info("Resumed")


class PausedState(GameState):
    def on_enter(self) -> None:
        dbg.info("PAUSED - P to resume")

    def on_update(self, dt: float, frame: int) -> None:
        stats.tick(dt)
        hud.clear(0)
        hud.panel(362, 300, 300, 80, 0x111133, 0x4444AA)
        hud.text(400, 320, "PAUSED", 0xFFFFFF, 2)
        hud.text(390, 355, "Press P to resume", 0x888888, 1)
        hud.present()
        if loop.input.is_key_down(KEY_P) == 1:
            self.machine.pop()
        if loop.input.is_key_down(KEY_ESCAPE) == 1:
            window.close()

    def on_exit(self) -> None:
        dbg.info("Unpaused")


sm = StateMachine()
sm.push(PlayingState())
sm.connect(loop)

loop.run()
hud.close()
