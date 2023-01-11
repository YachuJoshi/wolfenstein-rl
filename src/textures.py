import pygame

# textures
background = pygame.image.load("./images/textures/background.png").convert()
walls = pygame.image.load("./images/textures/walls.png").convert()
textures = {
    "S": walls.subsurface(0, 0, 64, 64),
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
