from src.textures import enemy


class Enemy:
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        static: bool = True,
        is_attacking: bool = False,
        distance_threshold: int = 300,
    ):
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
        self.attack_index = 0
        self.walk_index = 0
        self.left_index = 0
        self.right_index = 0
        self.is_attacking = is_attacking
        self.distance_threshold = distance_threshold
        self.image = enemy.subsurface(0, 0, 64, 64)
        self.death_animation_list = [
            enemy.subsurface(frame * 64, 5 * 64, 64, 64) for frame in range(1, 5)
        ]
        self.attack_animation_list = [
            enemy.subsurface(frame * 64, 6 * 64, 64, 64) for frame in range(1, 3)
        ]
        self.diagonal_walking_animation_list = [
            enemy.subsurface(0, frame * 64, 64, 64) for frame in range(1, 5)
        ]
        self.left_walking_animation_list = [
            enemy.subsurface(128, frame * 64, 64, 64) for frame in range(1, 5)
        ]
        self.right_walking_animation_list = [
            enemy.subsurface(384, frame * 64, 64, 64) for frame in range(1, 5)
        ]

    def __str__(self) -> str:
        return f"""
                Enemy: {self.id} -> {self.dead}
                """

    def __repr__(self) -> str:
        return f"""
                Enemy: {self.id} -> {self.dead}
                """
