class PointShape:
    def __init__(self, points: list):
        self.points = points


# Holds non-ephemeral object definitions (ephemerals aren't stored).
class ObjectDefinitions:
    def __init__(self):
        self.definitions = {}

    def add_definition(self, name: str, shape: PointShape):
        self.definitions[name] = shape

    def get_definition(self, name: str) -> PointShape:
        return self.definitions.get(name, None)
