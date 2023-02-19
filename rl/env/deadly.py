import gym
import cv2
import pygame
import numpy as np
from src.base import *
from src.screen import window, clock
from src.enemy import Enemy
from collections import namedtuple
from gym.spaces import Box, Discrete
from src.textures import background, gun, textures
from math import sin, cos, sqrt, atan2, degrees, dist


Point = namedtuple("Point", ("x", "y"))
INITIAL_ANGLE = 1.55
GEM_POSITION = {"x": 1260.0, "y": 182.0}
MAP, MAP_SIZE, MAP_RANGE, MAP_SPEED = get_map_details("DEADLY")


class WolfensteinDeadlyCorridorEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 120,
    }

    def __init__(self, render_mode=None):
        super(WolfensteinDeadlyCorridorEnv, self).__init__()

        shape = (100, 160, 1)
        self.observation_space = Box(0, 255, shape=(shape), dtype=np.uint8)
        self.action_space = Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window, self.clock = window, clock
        self.enemies = []
        self.player_x = 86.0
        self.player_y = 160.0
        self.player_angle = INITIAL_ANGLE
        self.ammo_count = 50
        self.player_health = 100
        self.enemy_death_count = 0
        self.zbuffer = []

    def _get_obs(self):
        return self._render_frame()

    def _get_info(self):
        return {}

    def _enemy_hit(self, enemy: Enemy) -> None:
        enemy.dead = True
        enemy.dx = 0
        self.reward += 100
        self.enemy_death_count += 1

    def _transform_image(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        reshaped = np.reshape(resized, (100, 160, 1))
        return reshaped

    def reset(self):
        self.reward = 0
        self.zbuffer = []
        self.done = False
        self.ammo_count = 50
        self.player_x = 86.0
        self.player_y = 160.0
        self.player_health = 100
        self.enemy_death_count = 0
        self.player_angle = INITIAL_ANGLE
        self.enemies = [
            Enemy(id=1, x=304.0, y=98.0, is_attacking=True),
            Enemy(id=2, x=450.0, y=236.0, is_attacking=True),
            Enemy(id=3, x=680.0, y=98.0, is_attacking=True),
            Enemy(id=4, x=850.0, y=236.0, is_attacking=True),
            Enemy(id=5, x=1200.0, y=98.0, is_attacking=True),
            Enemy(id=6, x=1250.0, y=236.0, is_attacking=True),
        ]

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        self.reward = 0
        self.done = False

        offset_x = sin(INITIAL_ANGLE) * MAP_SPEED
        offset_y = cos(INITIAL_ANGLE) * MAP_SPEED
        distance_thresh_x = 20 if offset_x > 0 else -20
        distance_thresh_y = 20 if offset_y > 0 else -20

        target_x = int(self.player_y / MAP_SCALE) * MAP_SIZE + int(
            (self.player_x + offset_x + distance_thresh_x) / MAP_SCALE
        )
        target_y = int(
            (self.player_y + offset_y + distance_thresh_y) / MAP_SCALE
        ) * MAP_SIZE + int(self.player_x / MAP_SCALE)

        # 0 -> Turn Left
        # 1 -> Turn Right
        # 2 -> Move Forward in MAP
        # 3 -> Move Backward in MAP
        # 4 -> Attack

        if action == 0:
            self.player_angle += 0.04

        elif action == 1:
            self.player_angle -= 0.04

        elif action == 2 and self.player_x < 1265.0:
            if MAP[target_x] in " e":
                self.player_x += offset_x

        elif action == 3 and self.player_x > 86.0:
            if MAP[target_x] in " e":
                self.player_x -= offset_x

        elif action == 4:
            if gun["animation"] == False:
                gun["animation"] = True
                self.ammo_count -= 1

        self.player_angle %= DOUBLE_PI
        self.zbuffer = []

        # ray casting
        current_angle = self.player_angle + HALF_FOV
        start_x = int(self.player_x / MAP_SCALE) * MAP_SCALE
        start_y = int(self.player_y / MAP_SCALE) * MAP_SCALE
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
            for _ in range(0, MAP_RANGE, MAP_SCALE):
                vertical_depth = (target_x - self.player_x) / current_sin
                target_y = self.player_y + vertical_depth * current_cos
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
                        vertical_depth = (target_x - self.player_x) / current_sin
                        target_y = self.player_y + vertical_depth * current_cos
                    break
                target_x += direction_x * MAP_SCALE
            texture_offset_y = target_y

            # ray hits horizontal line
            target_y, direction_y = (
                (start_y + MAP_SCALE, 1) if current_cos >= 0 else (start_y, -1)
            )
            for _ in range(0, MAP_RANGE, MAP_SCALE):
                horizontal_depth = (target_y - self.player_y) / current_cos
                target_x = self.player_x + horizontal_depth * current_sin
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
                        horizontal_depth = (target_y - self.player_y) / current_cos
                        target_x = self.player_x + horizontal_depth * current_sin
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
            depth *= cos(self.player_angle - current_angle)

            wall_height = MAP_SCALE * 300 / (depth + 0.0001)
            if wall_height > 50000:
                wall_height = 50000
            wall_block = textures[texture].subsurface(
                (texture_offset - int(texture_offset / MAP_SCALE) * MAP_SCALE), 0, 1, 64
            )
            wall_block = pygame.transform.scale(wall_block, (1, abs(int(wall_height))))

            self.zbuffer.append(
                {
                    "image": wall_block,
                    "x": ray,
                    "y": int(HEIGHT / 2 - wall_height / 2),
                    "distance": depth,
                }
            )

            # increment angle
            current_angle -= STEP_ANGLE

        # position & scale enemies
        for enemy in self.enemies:
            distance_x = enemy.x - self.player_x
            distance_y = enemy.y - self.player_y
            distance = sqrt(distance_x * distance_x + distance_y * distance_y)
            sprite2player_angle = atan2(distance_x, distance_y)
            player2sprite_angle = sprite2player_angle - self.player_angle

            if distance_x < 0:
                player2sprite_angle += DOUBLE_PI
            if distance_x > 0 and degrees(player2sprite_angle) <= -180:
                player2sprite_angle += DOUBLE_PI
            if distance_x < 0 and degrees(player2sprite_angle) >= 180:
                player2sprite_angle -= DOUBLE_PI

            shift_rays = player2sprite_angle / STEP_ANGLE
            sprite_ray = CENTRAL_RAY - shift_rays

            sprite_height = (
                min(enemy.scale * MAP_SCALE * 300 / (distance + 0.0001), 400)
                if enemy.dead == True
                else enemy.scale * MAP_SCALE * 300 / (distance + 0.0001)
            )

            if not enemy.dead:
                if enemy.is_attacking and distance < enemy.distance_threshold:
                    enemy.image = enemy.attack_animation_list[
                        int(enemy.attack_index / 8)
                    ]
                    enemy.attack_index += 0.5

                    if enemy.attack_index > 15:
                        enemy.attack_index = 0

                    if np.random.rand() < 0.2:
                        self.player_health -= 0.4

                # Shoot & Enemy Dead
                if (
                    abs(shift_rays) < 20
                    and distance < enemy.distance_threshold
                    and gun["animation"]
                ):
                    enemy.image = enemy.death_animation_list[int(enemy.death_count / 8)]
                    enemy.death_count += 1

                    if enemy.death_count >= 16:
                        self._enemy_hit(enemy)

            # Enemy Dead
            else:
                enemy.image = enemy.death_animation_list[-1]

                if not enemy.dead:
                    self._enemy_hit(enemy)

            # Shoot & Enemy Dead
            if gun["shot_count"] > 16 and enemy.image in [
                enemy.death_animation_list[0],
                enemy.death_animation_list[1],
                enemy.death_animation_list[2],
            ]:
                try:
                    enemy.image = enemy.death_animation_list[
                        int(enemy.death_count / 8) + 2
                    ]
                    self._enemy_hit(enemy)

                except:
                    pass
                enemy.death_count += 1

                if enemy.death_count >= 32:
                    enemy.dead = True
                    enemy.death_count = 0

            sprite_image = pygame.transform.scale(
                enemy.image, (int(sprite_height), int(sprite_height))
            )

            self.zbuffer.append(
                {
                    "image": sprite_image,
                    "x": sprite_ray - int(sprite_height / 2),
                    "y": 100 - sprite_height * enemy.shift,
                    "distance": distance,
                }
            )
        player_coordinates = (self.player_x, self.player_y)
        gem_coordinates = (GEM_POSITION["x"], GEM_POSITION["y"])
        player_gem_distance = dist(player_coordinates, gem_coordinates)

        self.reward += 4 / player_gem_distance

        if not self.done:
            self.reward -= 1

        if self.player_health <= 0:
            self.reward -= 1000
            self.done = True

        if (self.enemy_death_count == len(self.enemies)) and (
            self.player_x >= GEM_POSITION["x"] - 20
        ):
            self.reward += 1000
            self.done = True

        if self.ammo_count == 0 and self.enemy_death_count < len(self.enemies):
            self.reward -= 500
            self.done = True

        observation = self._get_obs()
        reward = self.reward
        done = self.done
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def _get_rgb(self):
        observation = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
        return self._transform_image(observation)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        self.window.blit(background, (0, 0))

        self.zbuffer = sorted(self.zbuffer, key=lambda k: k["distance"], reverse=True)
        for item in self.zbuffer:
            self.window.blit(item["image"], (item["x"], item["y"]))

        # render gun / gun animation
        self.window.blit(gun["default"], (60, 20))
        if gun["animation"]:
            gun["animation"] = True
            self.window.blit(gun["shot"][int(gun["shot_count"] / 5)], (60, 20))
            gun["shot_count"] += 1
            if gun["shot_count"] >= 20:
                gun["shot_count"] = 0
                gun["animation"] = False

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return self._get_rgb()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
