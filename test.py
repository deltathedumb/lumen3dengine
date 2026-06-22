from pugtk import Window, PointShape, Vector2, color

window = Window(640, 480)
triangle = PointShape([
    Vector2(100, 100),
    Vector2(200, 50),
    Vector2(150, 200),
])

running = 1
while running:
    window.draw(triangle, color(255, 0, 0))
    running = window.update()

window.close()
