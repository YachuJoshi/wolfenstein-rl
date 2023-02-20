import pygame
from pygame import Surface
from pygame.time import Clock
from src.base import WIDTH, HEIGHT

# init pygame
pygame.init()
pygame.display.init()

pygame.display.set_caption("Wolfenstein 3D")
# pygame.mouse.set_visible(False)

window: Surface = pygame.display.set_mode((WIDTH, HEIGHT))
clock: Clock = Clock()
