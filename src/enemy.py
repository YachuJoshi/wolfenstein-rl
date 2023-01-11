from src.textures import enemy


class Enemy:
    def __init__(self, id: int, x: float, y: float, static=False):
        self.id = id
        self.x = x
        self.y = y
        self.shift = 0.4
        self.scale = 1.0
        self.type = "soldier"
        self.dead = False
        self.dx = 0 if static else 0.2
        self.dy = 0 if static else 0.2
        self.death_count = 0
        self.image = enemy.subsurface(0, 0, 64, 64)
        self.death_animation_list = [
            enemy.subsurface(frame * 64, 5 * 64, 64, 64) for frame in range(1, 5)
        ]

    def __str__(self):
        return f"""
                Enemy: {self.id} -> {self.dead}
                """

    def __repr__(self):
        return f"""
                Enemy: {self.id} -> {self.dead}
                """
