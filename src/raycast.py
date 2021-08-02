# packages
from math import sin, cos, pi
import pygame
import sys

# init pygame
pygame.init()
pygame.mouse.set_visible(False)
window = pygame.display.set_mode((320, 200), pygame.FULLSCREEN)
clock = pygame.time.Clock()

# map
MAP_SCALE = 5
MAP_SPEED = (MAP_SCALE / 2) / 10
MAP_SIZE = 20
'''
MAP = (
    '########'
    '#      #'
    '#      #'
    '#      #'
    '#      #'
    '#      #'
    '#      #'
    '########'
)
'''

MAP = (
    '####################'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '#                  #'
    '####################'
)


# player coordinates and view angle
player_x = MAP_SCALE + 1.0
player_y = MAP_SCALE + 1.0
player_angle = pi / 3

# game loop
while True:
    # get user input
    pygame.event.get()
    keys = pygame.key.get_pressed()
        
    # handle user input
    if keys[pygame.K_ESCAPE]: pygame.quit(); sys.exit(0);
    if keys[pygame.K_LEFT]: player_angle += 0.1
    if keys[pygame.K_RIGHT]: player_angle -= 0.1
    if keys[pygame.K_UP]:
        # move offset & collision detection
        offset_x = sin(player_angle) * MAP_SPEED
        offset_y = cos(player_angle) * MAP_SPEED
        target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int((player_x + offset_x) / MAP_SCALE)
        target_y = int((player_y + offset_y) / MAP_SCALE) * MAP_SIZE + int(player_x / MAP_SCALE)
        if MAP[target_x] == ' ': player_x += offset_x
        if MAP[target_y] == ' ': player_y += offset_y
    if keys[pygame.K_DOWN]:
        # move offset & collision detection
        offset_x = sin(player_angle) * MAP_SPEED
        offset_y = cos(player_angle) * MAP_SPEED
        target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int((player_x - offset_x) / MAP_SCALE)
        target_y = int((player_y - offset_y) / MAP_SCALE) * MAP_SIZE + int(player_x / MAP_SCALE)
        if MAP[target_x] == ' ': player_x -= offset_x
        if MAP[target_y] == ' ': player_y -= offset_y
    
    # draw background
    window.fill((100, 100, 100))
    
    # draw map (debug)
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            pygame.draw.rect(window,
            (50, 50, 50) if MAP[row * MAP_SIZE + col] != ' ' else (0, 0, 0),
            (col * MAP_SCALE, row * MAP_SCALE, MAP_SCALE, MAP_SCALE))
    pygame.draw.circle(window, (255, 0, 0), (int(player_x), int(player_y)), 2)
    pygame.draw.line(window, (255, 0, 0), (player_x, player_y), 
                    (player_x + sin(player_angle) * 5, player_y + cos(player_angle) * 5), 1)
    
    # fps
    clock.tick(60)

    # print FPS to screen
    font = pygame.font.SysFont('Monospace Regular', 30)
    fps_surface = font.render(str(int(clock.get_fps())), False, (255, 255, 255))
    window.blit(fps_surface, (296, 0))

    # update display
    pygame.display.flip()
