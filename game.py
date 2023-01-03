import pygame
import sys
from src.screen import window, clock
from src.base import *
from src.textures import *
from src.player import *
from src.utils import *
from math import sin, cos, sqrt, atan2, degrees


if __name__ == "__main__":
    soldier_dx = 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # get user input
        keys = pygame.key.get_pressed()

        # player move offset
        offset_x = sin(player_angle) * MAP_SPEED
        offset_y = cos(player_angle) * MAP_SPEED
        distance_thresh_x = 20 if offset_x > 0 else -20
        distance_thresh_y = 20 if offset_y > 0 else -20

        target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int(
            (player_x + offset_x + distance_thresh_x) / MAP_SCALE
        )
        target_y = int(
            (player_y + offset_y + distance_thresh_y) / MAP_SCALE
        ) * MAP_SIZE + int(player_x / MAP_SCALE)

        # handle user input
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit(0)

        if keys[pygame.K_LEFT]:
            player_angle += 0.03

        if keys[pygame.K_RIGHT]:
            player_angle -= 0.03

        if keys[pygame.K_w]:
            if MAP[target_x] in " e":
                player_x += offset_x
            if MAP[target_y] in " e":
                player_y += offset_y

        if keys[pygame.K_s]:
            target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int(
                (player_x - offset_x - distance_thresh_x) / MAP_SCALE
            )
            target_y = int(
                (player_y - offset_y - distance_thresh_y) / MAP_SCALE
            ) * MAP_SIZE + int(player_x / MAP_SCALE)

            if MAP[target_x] in " e":
                player_x -= offset_x
            if MAP[target_y] in " e":
                player_y -= offset_y

        if keys[pygame.K_a]:
            target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int(
                (player_x - offset_x - distance_thresh_x) / MAP_SCALE
            )

            # target_y = int(
            #     (player_y + offset_y + distance_thresh_y) / MAP_SCALE
            # ) * MAP_SIZE - int(player_x / MAP_SCALE)

            if MAP[target_x] in " e":
                player_x += offset_y

            if MAP[target_y] in " e":
                player_y -= offset_x

        if keys[pygame.K_d]:
            target_x = int(player_y / MAP_SCALE) * MAP_SIZE + int(
                (player_x + offset_x + distance_thresh_x) / MAP_SCALE
            )

            # target_y = int(
            #     (player_y + offset_y + distance_thresh_y) / MAP_SCALE
            # ) * MAP_SIZE - int(player_x / MAP_SCALE)

            if MAP[target_x] in " e":
                player_x -= offset_y
            if MAP[target_y] in " e":
                player_y += offset_x

        if keys[pygame.K_SPACE]:
            if MAP[target_x] in "Ee":
                pygame.time.wait(200)
                MAP[target_x] = chr(ord(MAP[target_x]) ^ 1 << 5)
            if MAP[target_y] in "Ee":
                pygame.time.wait(200)
                MAP[target_y] = chr(ord(MAP[target_y]) ^ 1 << 5)

        if keys[pygame.K_LCTRL]:
            if gun["animation"] == False:
                gun["animation"] = True

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
        texture_y = "S"
        texture_x = "S"

        for ray in range(WIDTH):
            current_sin = sin(current_angle)
            current_sin = current_sin if current_sin else 0.000001
            current_cos = cos(current_angle)
            current_cos = current_cos if current_cos else 0.000001

            # ray hits vertical line
            target_x, direction_x = (
                (start_x + MAP_SCALE, 1) if current_sin >= 0 else (start_x, -1)
            )
            for i in range(0, MAP_RANGE, MAP_SCALE):
                vertical_depth = (target_x - player_x) / current_sin
                target_y = player_y + vertical_depth * current_cos
                map_x = int(target_x / MAP_SCALE)
                map_y = int(target_y / MAP_SCALE)
                if current_sin <= 0:
                    map_x += direction_x
                target_square = map_y * MAP_SIZE + map_x
                if target_square not in range(len(MAP)):
                    break
                if MAP[target_square] not in " e":
                    texture_y = MAP[target_square] if MAP[target_square] != "T" else "I"
                    if MAP[target_square] == "E":
                        target_x += direction_x * 32
                        vertical_depth = (target_x - player_x) / current_sin
                        target_y = player_y + vertical_depth * current_cos
                    break
                target_x += direction_x * MAP_SCALE
            texture_offset_y = target_y

            # ray hits horizontal line
            target_y, direction_y = (
                (start_y + MAP_SCALE, 1) if current_cos >= 0 else (start_y, -1)
            )
            for i in range(0, MAP_RANGE, MAP_SCALE):
                horizontal_depth = (target_y - player_y) / current_cos
                target_x = player_x + horizontal_depth * current_sin
                map_x = int(target_x / MAP_SCALE)
                map_y = int(target_y / MAP_SCALE)
                if current_cos <= 0:
                    map_y += direction_y
                target_square = map_y * MAP_SIZE + map_x
                if target_square not in range(len(MAP)):
                    break
                if MAP[target_square] not in " e":
                    texture_x = MAP[target_square] if MAP[target_square] != "O" else "J"
                    if MAP[target_square] == "E":
                        target_y += direction_y * 32
                        horizontal_depth = (target_y - player_y) / current_cos
                        target_x = player_x + horizontal_depth * current_sin
                    break
                target_y += direction_y * MAP_SCALE
            texture_offset_x = target_x

            # calculate 3D projection
            texture_offset = (
                texture_offset_y
                if vertical_depth < horizontal_depth
                else texture_offset_x
            )
            texture = texture_y if vertical_depth < horizontal_depth else texture_x
            depth = (
                vertical_depth
                if vertical_depth < horizontal_depth
                else horizontal_depth
            )
            depth *= cos(player_angle - current_angle)
            wall_height = MAP_SCALE * 300 / (depth + 0.0001)
            if wall_height > 50000:
                wall_height = 50000
            wall_block = textures[texture].subsurface(
                (texture_offset - int(texture_offset / MAP_SCALE) * MAP_SCALE), 0, 1, 64
            )
            wall_block = pygame.transform.scale(wall_block, (1, abs(int(wall_height))))
            zbuffer.append(
                {
                    "image": wall_block,
                    "x": ray,
                    "y": int(HEIGHT / 2 - wall_height / 2),
                    "distance": depth,
                }
            )

            # increment angle
            current_angle -= STEP_ANGLE

        # position & scale sprites
        for sprite in sprites:
            sprite_x = sprite["x"] - player_x
            sprite_y = sprite["y"] - player_y
            sprite_distance = sqrt(sprite_x * sprite_x + sprite_y * sprite_y)
            sprite2player_angle = atan2(sprite_x, sprite_y)
            player2sprite_angle = sprite2player_angle - player_angle
            if sprite_x < 0:
                player2sprite_angle += DOUBLE_PI
            if sprite_x > 0 and degrees(player2sprite_angle) <= -180:
                player2sprite_angle += DOUBLE_PI
            if sprite_x < 0 and degrees(player2sprite_angle) >= 180:
                player2sprite_angle -= DOUBLE_PI
            shift_rays = player2sprite_angle / STEP_ANGLE
            sprite_ray = CENTRAL_RAY - shift_rays
            if (
                sprite["type"] in ["lamp", "light"]
                or sprite["type"] == "soldier"
                and sprite["dead"] == True
            ):
                sprite_height = min(
                    sprite["scale"] * MAP_SCALE * 300 / (sprite_distance + 0.0001), 400
                )
            else:
                sprite_height = (
                    sprite["scale"] * MAP_SCALE * 300 / (sprite_distance + 0.0001)
                )
            if sprite["type"] == "soldier":
                if not sprite["dead"]:

                    if sprite["x"] > 360 or sprite["x"] < 82:
                        soldier_dx *= -1

                    sprite["x"] += soldier_dx

                    if (
                        abs(shift_rays) < 20
                        and sprite_distance < 500
                        and gun["animation"]
                    ):
                        sprite["image"] = soldier_death[int(soldier_death_count / 8)]
                        soldier_death_count += 1
                        if soldier_death_count >= 16:
                            sprite["dead"] = True
                            soldier_death_count = 0
                            soldier_dx = 0
                            print("DEAD FROM ABOVE")
                else:
                    sprite["image"] = soldier_death[-1]
                if gun["shot_count"] > 16 and sprite["image"] in [
                    soldier_death[0],
                    soldier_death[1],
                    soldier_death[2],
                ]:
                    try:
                        sprite["image"] = soldier_death[
                            int(soldier_death_count / 8) + 2
                        ]
                        soldier_dx = 0
                        print("DEAD FROM BELOW 1234")
                    except:
                        pass
                    soldier_death_count += 1
                    if soldier_death_count >= 32:
                        sprite["dead"] = True
                        soldier_death_count = 0
                if not sprite["dead"] and sprite_distance <= 10:
                    player_x -= offset_x
                    player_y -= offset_y
            sprite_image = pygame.transform.scale(
                sprite["image"], (int(sprite_height), int(sprite_height))
            )
            zbuffer.append(
                {
                    "image": sprite_image,
                    "x": sprite_ray - int(sprite_height / 2),
                    "y": 100 - sprite_height * sprite["shift"],
                    "distance": sprite_distance,
                }
            )

        # render scene
        zbuffer = sorted(zbuffer, key=lambda k: k["distance"], reverse=True)
        for item in zbuffer:
            window.blit(item["image"], (item["x"], item["y"]))

        # render gun / gun animation
        window.blit(gun["default"], (60, 20))
        if gun["animation"]:
            gun["animation"] = True
            window.blit(gun["shot"][int(gun["shot_count"] / 5)], (60, 20))
            gun["shot_count"] += 1
            if gun["shot_count"] >= 20:
                gun["shot_count"] = 0
                gun["animation"] = False
            pygame.display.flip()

        if keys[pygame.K_TAB]:
            draw_minimap(window, sprites, player_x, player_y, player_angle)
        # if keys[pygame.K_f]:
        #     show_fps(window, clock)

        # fps
        clock.tick(60)

        # update display
        pygame.display.update()
