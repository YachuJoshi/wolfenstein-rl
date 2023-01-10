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
