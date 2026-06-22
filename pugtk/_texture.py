class Texture:
    """A flat width*height pixel buffer (0x00RRGGBB per pixel, row-major) --
    the texture-mapping system's only contract. Decoding an actual PNG (or
    any other format) into one of these is left to whoever is loading
    assets; this class only needs the decoded pixels, not the file format.

    Named Texture rather than Image: lumen's SDL backend already has its
    own internal Image class (a GPU texture handle), and asmpython's
    whole-program compiler merges same-named classes from different
    modules into one flat symbol table without actually checking they
    agree -- so reusing "Image" here silently corrupted unrelated
    type-checking elsewhere in the program instead of raising a clean
    "name collides" error.
    """

    width: int
    height: int
    pixels: list[int]

    def __init__(self, width: int, height: int, pixels: list[int]):
        self.width = width
        self.height = height
        self.pixels = pixels

    def get_pixel(self, x: int, y: int) -> int:
        cx: int = x
        cy: int = y
        if cx < 0:
            cx = 0
        if cx >= self.width:
            cx = self.width - 1
        if cy < 0:
            cy = 0
        if cy >= self.height:
            cy = self.height - 1
        return self.pixels[cy * self.width + cx]

    @staticmethod
    def solid(width: int, height: int, color: int):
        """A single flat color as a 1x1-equivalent texture -- mainly useful
        as a placeholder before a real texture is loaded."""
        pixels: list[int] = []
        total: int = width * height
        i: int = 0
        while i < total:
            pixels.append(color)
            i = i + 1
        return Texture(width, height, pixels)
