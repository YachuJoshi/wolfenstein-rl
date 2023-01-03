import pygame

# textures
background = pygame.image.load("./images/textures/background.png").convert()
walls = pygame.image.load("./images/textures/walls.png").convert()
textures = {
    "S": walls.subsurface(0, 0, 64, 64),
    "T": walls.subsurface(0, 0, 64, 64),
    "O": walls.subsurface(0, 0, 64, 64),
    "D": walls.subsurface(2 * 64, 2 * 64, 64, 64),
    "W": walls.subsurface(4 * 64, 3 * 64, 64, 64),
    "X": walls.subsurface(0, 2 * 64, 64, 64),
    "B": walls.subsurface(2 * 64, 5 * 64, 64, 64),
    "E": walls.subsurface(2 * 64, 16 * 64, 64, 64),
    "I": walls.subsurface(4 * 64, 16 * 64, 64, 64),
    "J": walls.subsurface(5 * 64, 16 * 64, 64, 64),
}

# sprites
enemy = pygame.image.load("./images/sprites/enemy.png").convert_alpha()
lamp = pygame.image.load("./images/sprites/greenlight.png").convert_alpha()
light = pygame.image.load("./images/sprites/floorlight.png").convert_alpha()
sprites = [
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 120,
    #     "y": 240,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    {
        "image": enemy.subsurface(0, 0, 64, 64),
        "x": 212,
        "y": 360,
        "shift": 0.4,
        "scale": 1.0,
        "type": "soldier",
        "dead": False,
    },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 150,
    #     "y": 500,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 270,
    #     "y": 700,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 250,
    #     "y": 720,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 850,
    #     "y": 400,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 850,
    #     "y": 600,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 550,
    #     "y": 750,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 850,
    #     "y": 750,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 550,
    #     "y": 940,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 850,
    #     "y": 940,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 1050,
    #     "y": 1100,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 1050,
    #     "y": 1300,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 1250,
    #     "y": 1100,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 1250,
    #     "y": 1300,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 700,
    #     "y": 1200,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 700,
    #     "y": 1300,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 600,
    #     "y": 1200,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {
    #     "image": enemy.subsurface(0, 0, 64, 64),
    #     "x": 600,
    #     "y": 1300,
    #     "shift": 0.4,
    #     "scale": 1.0,
    #     "type": "soldier",
    #     "dead": False,
    # },
    # {"image": lamp, "x": 1140, "y": 1250, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {
    #     "image": light,
    #     "x": 1140,
    #     "y": 1250,
    #     "shift": -0.1,
    #     "scale": 1.0,
    #     "type": "light",
    # },
    # {"image": lamp, "x": 230, "y": 160, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 230, "y": 160, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 230, "y": 460, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 230, "y": 460, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 330, "y": 710, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 330, "y": 710, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 580, "y": 740, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 580, "y": 740, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 1050, "y": 740, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 1050, "y": 740, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 850, "y": 420, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 850, "y": 420, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 600, "y": 160, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 600, "y": 160, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 1140, "y": 100, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 1140, "y": 100, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 1140, "y": 400, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {"image": light, "x": 1140, "y": 400, "shift": -0.1, "scale": 1.0, "type": "light"},
    # {"image": lamp, "x": 1140, "y": 1050, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {
    #     "image": light,
    #     "x": 1140,
    #     "y": 1050,
    #     "shift": -0.1,
    #     "scale": 1.0,
    #     "type": "light",
    # },
    # {"image": lamp, "x": 1140, "y": 1250, "shift": 0.7, "scale": 1.0, "type": "light"},
    # {
    #     "image": light,
    #     "x": 1140,
    #     "y": 1250,
    #     "shift": -0.1,
    #     "scale": 1.0,
    #     "type": "light",
    # },
]

# gun
gun = {
    "default": pygame.image.load("./images/sprites/gun_0.png").convert_alpha(),
    "shot": [
        pygame.image.load("./images/sprites/gun_0.png").convert_alpha(),
        pygame.image.load("./images/sprites/gun_1.png").convert_alpha(),
        pygame.image.load("./images/sprites/gun_2.png").convert_alpha(),
        pygame.image.load("./images/sprites/gun_2.png").convert_alpha(),
    ],
    "shot_count": 0,
    "animation": False,
}

# animation
soldier_death = [enemy.subsurface(frame * 64, 5 * 64, 64, 64) for frame in range(1, 5)]
soldier_death_count = 0