# packages
from math import sin, cos, pi, sqrt, atan2, degrees
import pygame
import sys

# screen
WIDTH = 320
HEIGHT = 200

# camera
FOV = pi / 3
HALF_FOV = FOV / 2
STEP_ANGLE = FOV / WIDTH
CENTRAL_RAY = int(WIDTH / 2) - 1
DOUBLE_PI = 2 * pi

# init pygame
pygame.init()
pygame.mouse.set_visible(False)
window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)#, pygame.FULLSCREEN
clock = pygame.time.Clock()

# map
MAP_SIZE = 22
MAP_SCALE = 64
MAP_RANGE = MAP_SIZE * MAP_SCALE
MAP_SPEED = (MAP_SCALE / 2) / 10
MAP = list(
    'SSSSSSSBBFBBBBBBBBBBBB'
    'S     SB             B'
    'S     CB   BBBBBB B  B'
    'S     SB       WB B  F'
    'SSTETSSSWWWW   WB BBBB'
    'S      SW      WB    B'
    'P     SSW      WB    B'
    'C     OBBBBB   WWB   B'
    'S     E   BB     B   B'
    'STETSSOB  FB     BTETS'
    'S     SB  BBBBB  BS  S'
    'S     SB         BS  S'
    'SSSSSSSBBBBBBBBBBBS  S'
    'DDDDDDDDDOSSSSSSSSO  S'
    'D        E        E  S'
    'D  DDDDDDOSSSSSSSSO  M'
    'D  DDXDXDXDDDS   SS  S'
    'D  D        DS       Y'
    'D  D        LS       S'
    'D           RS       M'
    'D  D        DS       S'
    'DDDDDXDXDXDDDDSPSCSPSS'
)

# player coordinates and view angle
player_x = MAP_SCALE + 20.0
player_y = MAP_SCALE + 20.0
player_angle = pi / 3

# textures
background = pygame.image.load('images/textures/background.png').convert()
walls = pygame.image.load('images/textures/walls.png').convert()
textures = {
    'S': walls.subsurface(0, 0, 64, 64),
    'T': walls.subsurface(0, 0, 64, 64),
    'O': walls.subsurface(0, 0, 64, 64),
    'D': walls.subsurface(2 * 64, 2 * 64, 64, 64),
    'W': walls.subsurface(4 * 64, 3 * 64, 64, 64),
    'X': walls.subsurface(0, 2 * 64, 64, 64),
    'P': pygame.image.load('images/textures/pylogo.png').convert(),
    'C': pygame.image.load('images/textures/cmk.png').convert(),
    'Y': pygame.image.load('images/textures/youtube.png').convert(),
    'M': pygame.image.load('images/textures/monkey.png').convert(),
    'B': walls.subsurface(2 * 64, 5 * 64, 64, 64),
    'L': pygame.image.load('images/textures/no_more_left.png').convert(),
    'R': pygame.image.load('images/textures/no_more_right.png').convert(),
    'F': pygame.image.load('images/textures/xyz.png').convert(),
    'E': walls.subsurface(2 * 64, 16 * 64, 64, 64),
    'I': walls.subsurface(4 * 64, 16 * 64, 64, 64),
    'J': walls.subsurface(5 * 64, 16 * 64, 64, 64)
}

# sprites
enemy = pygame.image.load('images/sprites/enemy.png').convert_alpha()
lamp = pygame.image.load('images/sprites/greenlight.png').convert_alpha()
light = pygame.image.load('images/sprites/floorlight.png').convert_alpha()
sprites = [
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 200, 'y': 400, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 150, 'y': 500, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 270, 'y': 700, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 250, 'y': 720, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 850, 'y': 400, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 850, 'y': 600, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 550, 'y': 750, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 850, 'y': 750, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 550, 'y': 940, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 850, 'y': 940, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 1050, 'y': 1100, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 1050, 'y': 1300, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 1250, 'y': 1100, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 1250, 'y': 1300, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 700, 'y': 1200, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 700, 'y':1300, 'shift':  0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 600, 'y': 1200, 'shift': 0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': enemy.subsurface(0, 0, 64, 64), 'x': 600, 'y':1300, 'shift':  0.4, 'scale': 1.0, 'type': 'soldier', 'dead': False},
    {'image': lamp, 'x': 1140, 'y': 1250, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 1140, 'y':1250, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 230, 'y': 160, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 230, 'y': 160, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 230, 'y': 460, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 230, 'y': 460, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 330, 'y': 710, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 330, 'y': 710, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 580, 'y': 740, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 580, 'y': 740, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 1050, 'y': 740, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 1050, 'y': 740, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 850, 'y': 420, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 850, 'y': 420, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 600, 'y': 160, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 600, 'y': 160, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 1140, 'y': 100, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 1140, 'y': 100, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 1140, 'y': 400, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 1140, 'y': 400, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 1140, 'y': 1050, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 1140, 'y':1050, 'shift': -0.1, 'scale': 1.0, 'type': 'light'},
    {'image': lamp, 'x': 1140, 'y': 1250, 'shift': 0.7, 'scale': 1.0, 'type': 'light'},
    {'image': light, 'x': 1140, 'y':1250, 'shift': -0.1, 'scale': 1.0, 'type': 'light'}
]

# gun
gun = {
    'default': pygame.image.load('images/sprites/gun_0.png').convert_alpha(),
    'shot': [
        pygame.image.load('images/sprites/gun_0.png').convert_alpha(),
        pygame.image.load('images/sprites/gun_1.png').convert_alpha(),
        pygame.image.load('images/sprites/gun_2.png').convert_alpha(),
        pygame.image.load('images/sprites/gun_2.png').convert_alpha()
    ],
    'shot_count': 0,
    'animation': False
}

# animation
soldier_death = [enemy.subsurface(frame * 64, 5 * 64, 64, 64) for frame in range(1, 5)]
soldier_death_count = 0

# game loop
while True:
    # get user input
    pygame.event.get()
    keys = pygame.key.get_pressed()
    
    # player move offset
    offset_x = sin(player_angle) * MAP_SPEED
    offset_y = cos(player_angle) * MAP_SPEED
    distance_thresh_x = 10 if offset_x > 0 else -10
    distance_thresh_y = 10 if offset_y > 0 else -10
    target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int((player_x + offset_x + distance_thresh_x) / MAP_SCALE)
    target_y = int((player_y + offset_y + distance_thresh_y) / MAP_SCALE) * MAP_SIZE + int(player_x / MAP_SCALE)

    # handle user input
    if keys[pygame.K_ESCAPE]: pygame.quit(); sys.exit(0);
    if keys[pygame.K_LEFT]: player_angle += 0.03
    if keys[pygame.K_RIGHT]: player_angle -= 0.03
    if keys[pygame.K_UP]:
        if MAP[target_x] in ' e': player_x += offset_x
        if MAP[target_y] in ' e': player_y += offset_y
    if keys[pygame.K_DOWN]:
        target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int((player_x - offset_x - distance_thresh_x) / MAP_SCALE)
        target_y = int((player_y - offset_y - distance_thresh_y) / MAP_SCALE) * MAP_SIZE + int(player_x / MAP_SCALE)
        if MAP[target_x] in ' e': player_x -= offset_x
        if MAP[target_y] in ' e': player_y -= offset_y
    if keys[pygame.K_SPACE]:
        if MAP[target_x] in 'Ee': pygame.time.wait(200); MAP[target_x] = chr(ord(MAP[target_x]) ^ 1 << 5)
        if MAP[target_y] in 'Ee': pygame.time.wait(200); MAP[target_y] = chr(ord(MAP[target_y]) ^ 1 << 5)
    if keys[pygame.K_LCTRL]:
        if gun['animation'] == False: gun['animation'] = True
    
    # get rid of negative angles
    player_angle %= DOUBLE_PI

    # draw background
    window.blit(background, (0, 0))
    
    # zbuffer
    zbuffer = []
    
    # ray casting
    current_angle = player_angle + HALF_FOV
    start_x = int(player_x / MAP_SCALE) * MAP_SCALE
    start_y = int(player_y / MAP_SCALE) * MAP_SCALE
    texture_y = 'S'; texture_x = 'S'
    for ray in range(WIDTH):    
        current_sin = sin(current_angle); current_sin = current_sin if current_sin else 0.000001
        current_cos = cos(current_angle); current_cos = current_cos if current_cos else 0.000001

        # ray hits vertical line
        target_x, direction_x = (start_x + MAP_SCALE, 1) if current_sin >= 0 else (start_x, -1)
        for i in range(0, MAP_RANGE, MAP_SCALE):
            vertical_depth = (target_x - player_x) / current_sin
            target_y = player_y + vertical_depth * current_cos
            map_x = int(target_x / MAP_SCALE)
            map_y = int(target_y / MAP_SCALE)
            if current_sin <= 0: map_x += direction_x
            target_square = map_y * MAP_SIZE + map_x
            if target_square not in range(len(MAP)): break
            if MAP[target_square] not in ' e':
                texture_y = MAP[target_square] if MAP[target_square] != 'T' else 'I'
                if MAP[target_square] == 'E':
                    target_x += direction_x * 32
                    vertical_depth = (target_x - player_x) / current_sin;
                    target_y = player_y + vertical_depth * current_cos
                break            
            target_x += direction_x * MAP_SCALE
        texture_offset_y = target_y

        # ray hits horizontal line
        target_y, direction_y = (start_y + MAP_SCALE, 1) if current_cos >= 0 else (start_y, -1)
        for i in range(0, MAP_RANGE, MAP_SCALE):
            horizontal_depth = (target_y - player_y) / current_cos
            target_x = player_x + horizontal_depth * current_sin
            map_x = int(target_x / MAP_SCALE)
            map_y = int(target_y / MAP_SCALE)
            if current_cos <= 0: map_y += direction_y
            target_square = map_y * MAP_SIZE + map_x
            if target_square not in range(len(MAP)): break
            if MAP[target_square] not in ' e':
                texture_x = MAP[target_square] if MAP[target_square] != 'O' else 'J'
                if MAP[target_square] == 'E':
                    target_y += direction_y * 32;
                    horizontal_depth = (target_y - player_y) / current_cos;
                    target_x = player_x + horizontal_depth * current_sin
                break                
            target_y += direction_y * MAP_SCALE
        texture_offset_x = target_x

        # calculate 3D projection
        texture_offset = texture_offset_y if vertical_depth < horizontal_depth else texture_offset_x
        texture = texture_y if vertical_depth < horizontal_depth else texture_x
        depth = vertical_depth if vertical_depth < horizontal_depth else horizontal_depth
        depth *= cos(player_angle - current_angle)
        wall_height = MAP_SCALE * 300 / (depth + 0.0001)
        if wall_height > 50000: wall_height = 50000;
        wall_block = textures[texture].subsurface((texture_offset - int(texture_offset / MAP_SCALE) * MAP_SCALE), 0, 1, 64)
        wall_block = pygame.transform.scale(wall_block, (1, abs(int(wall_height))))
        zbuffer.append({'image': wall_block, 'x': ray, 'y': int(HEIGHT / 2 - wall_height / 2), 'distance': depth})

        # increment angle
        current_angle -= STEP_ANGLE

    # position & scale sprites
    for sprite in sprites:
        sprite_x = sprite['x'] - player_x
        sprite_y = sprite['y'] - player_y
        sprite_distance = sqrt(sprite_x * sprite_x + sprite_y * sprite_y)
        sprite2player_angle = atan2(sprite_x, sprite_y)
        player2sprite_angle = sprite2player_angle - player_angle
        if sprite_x < 0: player2sprite_angle += DOUBLE_PI
        if sprite_x > 0 and degrees(player2sprite_angle) <= -180: player2sprite_angle += DOUBLE_PI
        if sprite_x < 0 and degrees(player2sprite_angle) >= 180: player2sprite_angle -= DOUBLE_PI
        shift_rays = player2sprite_angle / STEP_ANGLE        
        sprite_ray = CENTRAL_RAY - shift_rays
        if sprite['type'] in ['lamp', 'light'] or sprite['type'] == 'soldier' and sprite['dead'] == True:
            sprite_height = min(sprite['scale'] * MAP_SCALE * 300 / (sprite_distance + 0.0001), 400)
        else: sprite_height = sprite['scale'] * MAP_SCALE * 300 / (sprite_distance + 0.0001)
        if sprite['type'] == 'soldier':
            if not sprite['dead']:
                if abs(shift_rays) < 20 and sprite_distance < 500 and gun['animation']:
                    sprite['image'] = soldier_death[int(soldier_death_count / 8)]
                    soldier_death_count += 1
                    if soldier_death_count >= 16: sprite['dead'] = True; soldier_death_count = 0
            else: sprite['image'] = soldier_death[-1]
            if gun['shot_count'] > 16 and sprite['image'] in [soldier_death[0], soldier_death[1], soldier_death[2]]:
                try: sprite['image'] = soldier_death[int(soldier_death_count / 8) + 2]
                except: pass
                soldier_death_count += 1
                if soldier_death_count >= 32: sprite['dead'] = True; soldier_death_count = 0
            if not sprite['dead'] and sprite_distance <= 10: player_x -= offset_x; player_y -= offset_y
        sprite_image = pygame.transform.scale(sprite['image'], (int(sprite_height), int(sprite_height)))
        zbuffer.append({'image': sprite_image,'x': sprite_ray - int(sprite_height / 2),
                        'y': 100 - sprite_height * sprite['shift'], 'distance': sprite_distance})

    # render scene
    zbuffer = sorted(zbuffer, key=lambda k: k['distance'], reverse=True)
    for item in zbuffer:
        window.blit(item['image'], (item['x'], item['y']))
    
    # render gun / gun animation
    window.blit(gun['default'], (60, 20))
    if gun['animation']:
        gun['animation'] = True
        window.blit(gun['shot'][int(gun['shot_count'] / 5)], (60, 20))
        gun['shot_count'] += 1
        if gun['shot_count'] >= 20: gun['shot_count'] = 0; gun['animation'] = False;
        pygame.display.flip()        

    # draw map (debug)
    if keys[pygame.K_TAB]:
        for row in range(MAP_SIZE):
            for col in range(MAP_SIZE):
                pygame.draw.rect(window,
                (100, 100, 100) if MAP[row * MAP_SIZE + col] != ' ' else (200, 200, 200),
                (col * 5, row * 5, 5, 5))
        pygame.draw.circle(window, (255, 0, 0), (int((player_x / MAP_SCALE) * 5), int((player_y / MAP_SCALE) * 5)), 2)
        pygame.draw.line(window, (255, 0, 0), ((player_x / MAP_SCALE) * 5, (player_y / MAP_SCALE) * 5), 
                        ((player_x / MAP_SCALE) * 5 + sin(player_angle) * 5, (player_y / MAP_SCALE) * 5 + cos(player_angle) * 5), 1)
        for sprite in sprites:
            if sprite['type'] == 'soldier' and not sprite['dead']:
                pygame.draw.circle(window, (0, 0, 255), (int((sprite['x'] / MAP_SCALE) * 5), int((sprite['y'] / MAP_SCALE) * 5)), 2)

    # fps
    clock.tick(60)

    # print FPS to screen
    font = pygame.font.SysFont('Monospace Regular', 30)
    fps_surface = font.render('FPS: ' + str(int(clock.get_fps())), False, (255, 0, 0))
    if keys[pygame.K_f]: window.blit(fps_surface, (120, 0))

    # update display
    pygame.display.flip()
