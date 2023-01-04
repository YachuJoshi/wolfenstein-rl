from src.textures import enemy


class Enemy:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.shift = 0.4
        self.scale = 1.0
        self.type = "soldier"
        self.dead = False
        self.dx = 0.2
        self.dy = 0.2
        self.death_count = 0
        self.image = enemy.subsurface(0, 0, 64, 64)
