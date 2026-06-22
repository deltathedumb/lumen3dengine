import math

from ._vector import Vector3
from ._window import shade, mix_colors
from ._texture import Texture


class ShaderContext:
    normal: Vector3
    world_pos: Vector3
    # Position in the mesh's own local/model space (before the model
    # matrix's rotation/translation). A procedural pattern keyed off
    # world_pos is fixed relative to the world, so it visibly slides/shears
    # across a face as the mesh rotates underneath it -- not how a real
    # texture behaves. Keying off local_pos instead makes the pattern rotate
    # together with the mesh, since it doesn't change as the mesh moves.
    local_pos: Vector3
    camera_pos: Vector3
    light_dir: Vector3
    light_color: int
    ambient: float
    specular: float
    shininess: float
    base_color: int
    # texture is set once per face (Renderer3D.render_solid_per_pixel only
    # -- render_solid has no per-pixel UV to sample, so it passes a 1x1
    # placeholder and tex_u/tex_v of 0.5, 0.5). tex_u/tex_v are the actual
    # per-pixel interpolated UV when there genuinely is one.
    texture: Texture
    tex_u: float
    tex_v: float
    # Only meaningful for per-pixel shading (Renderer3D.render_solid_per_pixel);
    # 0, 0 for per-face shading (Renderer3D.render_solid), since there's no
    # single pixel a face-level call corresponds to.
    screen_x: int
    screen_y: int

    def __init__(
        self,
        normal: Vector3,
        world_pos: Vector3,
        local_pos: Vector3,
        camera_pos: Vector3,
        light_dir: Vector3,
        light_color: int,
        ambient: float,
        specular: float,
        shininess: float,
        base_color: int,
        texture: Texture,
        tex_u: float,
        tex_v: float,
        screen_x: int,
        screen_y: int,
    ):
        self.normal = normal
        self.world_pos = world_pos
        self.local_pos = local_pos
        self.camera_pos = camera_pos
        self.light_dir = light_dir
        self.light_color = light_color
        self.ambient = ambient
        self.specular = specular
        self.shininess = shininess
        self.base_color = base_color
        self.texture = texture
        self.tex_u = tex_u
        self.tex_v = tex_v
        self.screen_x = screen_x
        self.screen_y = screen_y


def unlit_shader(ctx: ShaderContext) -> int:
    """No lighting at all -- always returns the face's base color."""
    return ctx.base_color


def lambert_shader(ctx: ShaderContext) -> int:
    """Flat Lambertian shading: brightness from normal-vs-light alignment,
    with an ambient floor so unlit faces aren't pure black."""
    intensity: float = ctx.normal.dot(ctx.light_dir)
    factor: float = ctx.ambient + (1.0 - ctx.ambient) * intensity
    return shade(ctx.base_color, factor)


def phong_shader(ctx: ShaderContext) -> int:
    """Lambertian diffuse + a Blinn-Phong specular highlight tinted by
    light_color. Needs ctx.specular (highlight strength, 0..1) and
    ctx.shininess (higher = smaller/sharper highlight) set on the
    Renderer3D, since those aren't part of the base lambert model."""
    view_dir: Vector3 = (ctx.camera_pos - ctx.world_pos).normalized()
    half_raw: Vector3 = ctx.light_dir + view_dir
    half_vec: Vector3 = half_raw.normalized()

    diffuse_intensity: float = ctx.normal.dot(ctx.light_dir)
    diffuse_factor: float = ctx.ambient + (1.0 - ctx.ambient) * diffuse_intensity

    spec_dot: float = ctx.normal.dot(half_vec)
    if spec_dot < 0.0:
        spec_dot = 0.0
    spec_power: float = math.pow(spec_dot, ctx.shininess)
    spec_factor: float = spec_power * ctx.specular

    diffuse_color: int = shade(ctx.base_color, diffuse_factor)
    return mix_colors(diffuse_color, ctx.light_color, 1.0, spec_factor)


def checker_shader(ctx: ShaderContext) -> int:
    """A per-pixel-only effect: a 3D checkerboard pattern from local_pos,
    lit the same way as lambert_shader. Used with Renderer3D.render_solid()
    this would just compute one cell's color for the whole flat face --
    the point of render_solid_per_pixel() is that ctx.local_pos genuinely
    varies pixel to pixel within a single face, so the checker pattern
    actually shows up.

    Keyed off local_pos rather than world_pos so the pattern stays glued to
    the mesh's own surface as it rotates, like a real texture, instead of
    sliding across the face (a pattern fixed in world space)."""
    cell_size: float = 0.5
    cx: int = math.floor(ctx.local_pos.x / cell_size)
    cy: int = math.floor(ctx.local_pos.y / cell_size)
    cz: int = math.floor(ctx.local_pos.z / cell_size)
    parity: int = (cx + cy + cz) % 2

    intensity: float = ctx.normal.dot(ctx.light_dir)
    factor: float = ctx.ambient + (1.0 - ctx.ambient) * intensity

    if parity == 0:
        return shade(ctx.base_color, factor)
    return shade(ctx.base_color, factor * 0.25)


def textured_shader(ctx: ShaderContext) -> int:
    """Samples ctx.texture at ctx.tex_u/ctx.tex_v (each genuinely varying
    per pixel, perspective-correct, under render_solid_per_pixel) instead
    of using a flat base_color, then lights it the same way as
    lambert_shader."""
    tex: Texture = ctx.texture
    px: int = int(ctx.tex_u * float(tex.width))
    py: int = int(ctx.tex_v * float(tex.height))
    base: int = tex.get_pixel(px, py)

    intensity: float = ctx.normal.dot(ctx.light_dir)
    factor: float = ctx.ambient + (1.0 - ctx.ambient) * intensity
    return shade(base, factor)
