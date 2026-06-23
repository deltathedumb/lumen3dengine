import math

from ._vector import Vector3


class PointLight:
    """A local light source for GLRenderer3D's multi-light pass (up to
    GLRenderer3D.max_lights of these per scene, see _gl_shaders.py's
    pointLight* uniform arrays). Point and spot are the same struct: a
    spot is a point light with a direction + cone half-angle (cutoff);
    leave spot_cutoff at its default (-1.0) for an omnidirectional point
    light, the shader checks `cutoff > -1.0` to decide which it is.

    Doesn't cast shadows -- only the single directional light
    (GLRenderer3D.light_dir) has a shadow map; see _gl_shaders.py's module
    docstring for why local-light shadows are out of scope for now.
    """

    position: Vector3
    color: Vector3
    range: float
    spot_direction: Vector3
    spot_cutoff: float

    def __init__(
        self,
        position: Vector3,
        color: Vector3,
        light_range: float,
        spot_direction: Vector3,
        spot_cutoff_deg: float,
    ):
        """spot_cutoff_deg <= -1.0 (e.g. PointLight.omni()'s -1.0) means
        an omnidirectional point light -- spot_direction is then unused
        but still required (pass Vector3(0.0, -1.0, 0.0) or any value)."""
        self.position = position
        self.color = color
        self.range = light_range
        self.spot_direction = spot_direction
        if spot_cutoff_deg <= -1.0:
            self.spot_cutoff = -1.0
        else:
            half_angle_rad: float = spot_cutoff_deg * math.pi / 180.0
            self.spot_cutoff = math.cos(half_angle_rad)

    @staticmethod
    def omni(position: Vector3, color: Vector3, light_range: float):
        """Convenience constructor for the common case (omnidirectional
        point light, no spot cone) -- avoids every caller having to spell
        out an unused spot_direction/spot_cutoff_deg pair."""
        return PointLight(position, color, light_range, Vector3(0.0, -1.0, 0.0), -1.0)
