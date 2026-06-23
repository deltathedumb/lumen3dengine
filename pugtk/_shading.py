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
    # 0 (dielectric -- plastic/wood/stone-like) to 1 (metallic, e.g. bare
    # steel/gold). Only consumed by pbr_shader()/pbr_rim_shader(); every
    # other shader here ignores it. Carried on Renderer3D (set
    # renderer.metallic) the same way specular/shininess already are,
    # rather than per-face, since per-face material variation isn't
    # plumbed through SceneObject/colors yet.
    metallic: float
    # 0 (mirror-smooth, a tight specular highlight) to 1 (fully rough, no
    # visible highlight at all). Only consumed by pbr_shader()/pbr_rim_shader().
    roughness: float

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
        metallic: float = 0.0,
        roughness: float = 0.5,
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
        self.metallic = metallic
        self.roughness = roughness


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


def _ggx_distribution(n_dot_h: float, alpha: float) -> float:
    """Trowbridge-Reitz/GGX normal distribution function: how concentrated
    the surface's microfacet normals are around the half-vector. Lower
    alpha (smoother/less rough) -> a tighter, brighter highlight; higher
    alpha -> a wider, dimmer one. This is the same NDF real-time PBR
    pipelines (Unreal, Unity's HDRP/URP, Godot) use, not a stylized
    approximation."""
    alpha2: float = alpha * alpha
    d: float = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0
    denom: float = math.pi * d * d
    if denom < 0.0001:
        denom = 0.0001
    return alpha2 / denom


def _schlick_fresnel(cos_theta: float, f0: float) -> float:
    """Schlick's approximation of the Fresnel term: reflectance rises
    toward 1.0 (full mirror) at grazing angles (cos_theta near 0) for any
    material, even ones that look mostly non-reflective head-on -- the same
    reason still water looks reflective at a shallow viewing angle but
    transparent looking straight down."""
    ct: float = cos_theta
    if ct < 0.0:
        ct = 0.0
    if ct > 1.0:
        ct = 1.0
    inv_ct: float = 1.0 - ct
    return f0 + (1.0 - f0) * math.pow(inv_ct, 5.0)


def _smith_geometry(n_dot_v: float, n_dot_l: float, alpha: float) -> float:
    """Smith's geometry/shadowing-masking term: approximates microfacets
    blocking each other's view of the light (shadowing) or the camera
    (masking) at grazing angles, which is why rough surfaces look dimmer
    near their silhouette than a naive Lambert/Phong model predicts."""
    k: float = (alpha + 1.0) * (alpha + 1.0) / 8.0
    g_v: float = n_dot_v / (n_dot_v * (1.0 - k) + k)
    g_l: float = n_dot_l / (n_dot_l * (1.0 - k) + k)
    return g_v * g_l


def pbr_shader(ctx: ShaderContext) -> int:
    """Cook-Torrance microfacet specular (GGX distribution + Schlick
    Fresnel + Smith geometry term) combined with a metallic-workflow
    diffuse term, the same lighting model real-time PBR engines (Unreal,
    Unity, Godot) use for their default "standard" material -- as opposed
    to lambert_shader/phong_shader's much cheaper, non-physically-based
    approximations.

    Needs ctx.metallic (0 = dielectric, 1 = metal) and ctx.roughness (0 =
    mirror-smooth, 1 = fully matte), both set via Renderer3D.metallic /
    Renderer3D.roughness (same per-renderer-not-per-face pattern
    specular/shininess already use for phong_shader)."""
    n_dot_l: float = ctx.normal.dot(ctx.light_dir)
    if n_dot_l < 0.0:
        n_dot_l = 0.0

    view_dir: Vector3 = (ctx.camera_pos - ctx.world_pos).normalized()
    n_dot_v: float = ctx.normal.dot(view_dir)
    if n_dot_v < 0.01:
        n_dot_v = 0.01

    half_vec: Vector3 = (ctx.light_dir + view_dir).normalized()
    n_dot_h: float = ctx.normal.dot(half_vec)
    if n_dot_h < 0.0:
        n_dot_h = 0.0
    v_dot_h: float = view_dir.dot(half_vec)

    roughness: float = ctx.roughness
    if roughness < 0.04:
        roughness = 0.04
    alpha: float = roughness * roughness

    # Dielectrics reflect ~4% of light head-on regardless of base color
    # (the standard PBR convention); metals tint their *specular*
    # reflection by their own base color instead of staying neutral white,
    # which is the visual signature that reads as "metal" vs. "plastic".
    f0: float = 0.04 + (1.0 - 0.04) * ctx.metallic

    d_term: float = _ggx_distribution(n_dot_h, alpha)
    f_term: float = _schlick_fresnel(v_dot_h, f0)
    g_term: float = _smith_geometry(n_dot_v, n_dot_l, alpha)

    spec_denom: float = 4.0 * n_dot_v * n_dot_l + 0.0001
    spec_strength: float = (d_term * f_term * g_term) / spec_denom
    # A single directional light (no area/solid angle to integrate the GGX
    # peak over, unlike a real area light or image-based environment
    # lighting) makes spec_strength genuinely unbounded as n_dot_h -> 1 --
    # at roughness 0.3 it already exceeds 1.0 well before the highlight's
    # exact center. Clamping caps the highlight's brightness instead of
    # letting it blow the color channels out to flat white.
    if spec_strength > 4.0:
        spec_strength = 4.0

    base_r: int = (ctx.base_color >> 16) & 0xFF
    base_g: int = (ctx.base_color >> 8) & 0xFF
    base_b: int = ctx.base_color & 0xFF

    # Energy conservation: light that goes into the specular lobe (scaled
    # by f_term, the Fresnel reflectance) can't also be diffusely
    # scattered, and a pure metal has no diffuse term at all (all incoming
    # light is either specularly reflected or absorbed).
    diffuse_factor: float = (1.0 - f_term) * (1.0 - ctx.metallic) * n_dot_l
    ambient_factor: float = ctx.ambient

    out_r: float = base_r * (ambient_factor + diffuse_factor) + 255.0 * spec_strength * n_dot_l
    out_g: float = base_g * (ambient_factor + diffuse_factor) + 255.0 * spec_strength * n_dot_l
    out_b: float = base_b * (ambient_factor + diffuse_factor) + 255.0 * spec_strength * n_dot_l

    r: int = int(out_r)
    g: int = int(out_g)
    b: int = int(out_b)
    if r > 255:
        r = 255
    if g > 255:
        g = 255
    if b > 255:
        b = 255
    if r < 0:
        r = 0
    if g < 0:
        g = 0
    if b < 0:
        b = 0
    return (r << 16) | (g << 8) | b


def fresnel_rim_shader(ctx: ShaderContext) -> int:
    """Lambert diffuse plus a Fresnel-driven rim/edge glow tinted by
    light_color -- surfaces brighten near their silhouette (where the
    surface normal is near-perpendicular to the view direction), the same
    grazing-angle brightening real materials show (a sphere's edge looking
    brighter than its center under even lighting) and a common stylized
    "energy shield" / backlit-edge look in games that don't need a full
    PBR pipeline."""
    view_dir: Vector3 = (ctx.camera_pos - ctx.world_pos).normalized()
    n_dot_v: float = ctx.normal.dot(view_dir)
    if n_dot_v < 0.0:
        n_dot_v = 0.0
    if n_dot_v > 1.0:
        n_dot_v = 1.0
    inv_n_dot_v: float = 1.0 - n_dot_v
    rim: float = math.pow(inv_n_dot_v, 3.0)

    intensity: float = ctx.normal.dot(ctx.light_dir)
    factor: float = ctx.ambient + (1.0 - ctx.ambient) * intensity
    diffuse: int = shade(ctx.base_color, factor)
    return mix_colors(diffuse, ctx.light_color, 1.0, rim)
