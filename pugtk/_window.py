import lumen

from ._shapes import PointShape


def color(r: int, g: int, b: int) -> int:
    # Returns 0x00RRGGBB
    return (r << 16) | (g << 8) | b


def shade(base_color: int, factor: float) -> int:
    """Scales a 0x00RRGGBB color's channels by factor (clamped to [0, 1])."""
    f: float = factor
    if f < 0.0:
        f = 0.0
    if f > 1.0:
        f = 1.0
    r: int = (base_color >> 16) & 0xFF
    g: int = (base_color >> 8) & 0xFF
    b: int = base_color & 0xFF
    r2: int = int(r * f)
    g2: int = int(g * f)
    b2: int = int(b * f)
    return (r2 << 16) | (g2 << 8) | b2


def mix_colors(c1: int, c2: int, f1: float, f2: float) -> int:
    """Per-channel clamped sum of c1*f1 + c2*f2 -- an additive blend, e.g.
    for layering a specular highlight (c2, scaled by intensity) on top of
    a diffuse base color (c1, scaled by 1.0)."""
    r1: int = (c1 >> 16) & 0xFF
    g1: int = (c1 >> 8) & 0xFF
    b1: int = c1 & 0xFF
    r2: int = (c2 >> 16) & 0xFF
    g2: int = (c2 >> 8) & 0xFF
    b2: int = c2 & 0xFF
    r: int = int(r1 * f1 + r2 * f2)
    g: int = int(g1 * f1 + g2 * f2)
    b: int = int(b1 * f1 + b2 * f2)
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


class Window:
    pixbuf: lumen.PixelBuffer
    canvas: lumen.Canvas

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.pixbuf = lumen.PixelBuffer(width, height)
        self.canvas = lumen.Canvas(f"pugtk_canvas_{id(self)}", width, height)
        self.canvas.clear()  # Initialize the canvas with a clear state
        self.pixbuf.clear()  # Initialize the pixel buffer with a clear state

    def clear(self):
        self.pixbuf.clear()

    def put_pixel(self, x: int, y: int, color: int):
        self.pixbuf.set(x, y, color)

    def get_pixel(self, x: int, y: int) -> int:
        return self.pixbuf.get(x, y)

    def update(self) -> int:
        # Mirror the pixel buffer to the canvas, then present + pump events
        self.canvas.blit_pixels(self.pixbuf, 0, 0)
        return self.canvas.update()

    def close(self):
        self.canvas.close()

    def draw(self, shape: PointShape, color: int):
        """Universal drawing entry point.

        2 points draws a line (Bresenham). 3+ points fills the polygon
        (scanline rasterization). Fewer than 2 points is a no-op.
        """
        points = shape.points
        if len(points) < 2:
            return

        if len(points) == 2:
            start = points[0]
            end = points[1]

            distance_x = abs(end.x - start.x)
            distance_y = abs(end.y - start.y)

            step_dir_x = 1 if end.x > start.x else -1
            step_dir_y = 1 if end.y > start.y else -1

            error = distance_x - distance_y

            curr_x = start.x
            curr_y = start.y

            while True:
                self.put_pixel(curr_x, curr_y, color)

                if curr_x == end.x and curr_y == end.y:
                    break

                double_error = 2 * error
                if double_error > -distance_y:
                    error -= distance_y
                    curr_x += step_dir_x
                if double_error < distance_x:
                    error += distance_x
                    curr_y += step_dir_y
            return

        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)

        for y in range(min_y, max_y + 1):
            intersections: list[int] = []

            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]

                if p1.y == p2.y:
                    continue

                edge_y_min = min(p1.y, p2.y)
                edge_y_max = max(p1.y, p2.y)

                # Half-open [edge_y_min, edge_y_max) edge inclusion: a vertex
                # shared by two edges (like the bend at (100,100) above)
                # would otherwise be counted by BOTH edges at that scanline,
                # producing 3 intersections instead of 2 and pairing into a
                # degenerate 1px sliver instead of merging into the real
                # span. Excluding each edge's upper endpoint fixes that, but
                # would also drop the polygon's bottommost row entirely
                # (every edge touching it has it as an upper endpoint) -- so
                # the last scanline keeps the inclusive bound to still draw
                # that final vertex point.
                if y == max_y:
                    in_range = edge_y_min <= y <= edge_y_max
                else:
                    in_range = edge_y_min <= y < edge_y_max

                if in_range:
                    # Integer cross-multiplication instead of float division:
                    # asmpython's int/int-to-float promotion truncates to
                    # integer division on the first dynamic execution of a
                    # conditionally-reached division inside nested loops (a
                    # confirmed compiler bug), giving garbage intersection
                    # x-values. This is mathematically equivalent to
                    # int(p1.x + (y-p1.y)/(p2.y-p1.y) * (p2.x-p1.x)).
                    numer = (y - p1.y) * (p2.x - p1.x)
                    den = p2.y - p1.y
                    x = p1.x + numer // den
                    intersections.append(x)

            if len(intersections) >= 2:
                intersections.sort()
                for j in range(0, len(intersections), 2):
                    if j + 1 < len(intersections):
                        x1 = intersections[j]
                        x2 = intersections[j + 1]
                        self.pixbuf.fill_rect(x1, y, x2 - x1 + 1, 1, color)
