import os
import cv2
import gym
import pygame
import numpy as np

from src.base import *
from src.screen import *
from src.font import font
from src.enemy import Enemy
from src.textures import background, gun, textures

from typing import Union, Tuple, Dict, Literal

from collections import namedtuple
from gym.spaces import Box, Discrete
from math import sin, cos, sqrt, atan2, degrees

FILE_PATH = "images/sprites"
bullet = pygame.image.load(os.path.join(FILE_PATH, "bullet.png")).convert_alpha()
heart = pygame.image.load(os.path.join(FILE_PATH, "heart.png")).convert_alpha()
head = pygame.image.load(os.path.join(FILE_PATH, "head.png")).convert_alpha()
bullet_rect = bullet.get_rect(topright=(WIDTH - 50, 10))
heart_rect = heart.get_rect(topleft=(10, 10))
head_rect = head.get_rect(midtop=(WIDTH / 2 - 14, 10))


Point = namedtuple("Point", ("x", "y"))
coordinates = [
    Point(95.0, 95.0),
    Point(540.0, 95.0),
    Point(95.0, 540.0),
    Point(540.0, 540.0),
]

TypeStep = Tuple[np.ndarray, float, bool, dict]
RenderMode = Union[Literal["human"], Literal["rgb_array"], None]

MAP, MAP_SIZE, MAP_RANGE, MAP_SPEED = get_map_details("DEFEND")


class WolfensteinDefendTheCenterEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 120,
    }

    def __init__(self, render_mode: RenderMode = None) -> None:
        super(WolfensteinDefendTheCenterEnv, self).__init__()

        shape = (100, 160, 1)
        self.observation_space = Box(0, 255, shape=(shape), dtype=np.uint8)
        self.action_space = Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window, self.clock = window, clock
        self.enemies = []
        self.player_x = 320.0
        self.player_y = 320.0
        self.player_angle = 1.5
        self.ammo_count = 100
        self.player_health = 100
        self.zbuffer = []
        self.kill_count = 0

    def _get_obs(self) -> np.ndarray:
        return self._render_frame()

    def _get_info(self) -> Dict[str, int]:
        return {"info": self.ammo_count}

    def _transform_image(self, observation: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        reshaped = np.reshape(resized, (100, 160, 1))
        return reshaped

    def _regenerate_enemies(self, index: int) -> None:
        self.enemies = list(filter(lambda enemy: (enemy.id != index), self.enemies))
        x, y = coordinates[index - 1]
        self.enemies.append(Enemy(id=index, x=x, y=y, static=False, is_defend=True))
        self.enemies = sorted(self.enemies, key=lambda enemy: enemy.id)

    def _enemy_hit(self, enemy: Enemy, index: int) -> None:
        self.kill_count += 1
        enemy.dead = True
        enemy.death_count = 0
        enemy.dx = 0
        enemy.dy = 0
        self._regenerate_enemies(index)

    def reset(self) -> np.ndarray:
        self.zbuffer = []
        self.kill_count = 0
        self.reward = 0
        self.done = False
        self.player_x = 320.0
        self.player_y = 320.0
        self.player_angle = 1.5
        self.player_health = 100
        self.ammo_count = 100
        self.enemies = [
            Enemy(id=index, x=x, y=y, static=False, is_defend=True)
            for index, (x, y) in enumerate(coordinates, start=1)
        ]

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action: int) -> TypeStep:
        self.reward = 0
        self.done = False

        # 0 -> Turn Left
        # 1 -> Turn Right
        # 2 -> Attack

        if action == 0:
            self.player_angle += 0.04

        elif action == 1:
            self.player_angle -= 0.04

        elif action == 2:
            if gun["animation"] == False and self.ammo_count > 0:
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
        for index, enemy in enumerate(self.enemies, start=1):
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
                if enemy.x < self.player_x and enemy.dx < 0:
                    enemy.dx *= 1
                elif enemy.x > self.player_x and enemy.dx > 0:
                    enemy.dx *= -1

                if enemy.y < self.player_y and enemy.dy < 0:
                    enemy.dy *= 1
                elif enemy.y > self.player_y and enemy.dy > 0:
                    enemy.dy *= -1

                enemy.x += enemy.dx
                enemy.y += enemy.dy

                enemy.image = enemy.diagonal_walking_animation_list[
                    int(enemy.walk_index / 8)
                ]
                enemy.walk_index += 1.5

                if enemy.walk_index > 8 * len(enemy.diagonal_walking_animation_list):
                    enemy.walk_index = 0

                # Shoot & Enemy Dead
                if abs(shift_rays) < 20 and distance < 280 and gun["animation"]:
                    self._enemy_hit(enemy, index)
                    self.reward += 1

                if distance <= 10:
                    self.player_health -= 25

                    # Remove that enemy & Add that enemy to the bounday
                    self._regenerate_enemies(index)

                    if self.player_health <= 0:
                        self.reward -= 1
                        self.done = True
            # Enemy Dead
            else:
                self._enemy_hit(enemy, index)
                enemy.image = enemy.death_animation_list[-1]

            # Shoot & Enemy Dead
            if gun["shot_count"] > 4 and enemy.image in [
                enemy.death_animation_list[0],
                enemy.death_animation_list[1],
                enemy.death_animation_list[2],
            ]:
                try:
                    self._enemy_hit(enemy, index)
                    # enemy.image = enemy.death_animation_list[
                    #     int(enemy.death_count / 8) + 2
                    # ]

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

        observation = self._get_obs()
        reward = self.reward
        done = self.done
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def _get_rgb(self) -> np.ndarray:
        observation = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
        return self._transform_image(observation)

    def render(self) -> Union[np.ndarray, None]:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.ndarray:
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

        self.window.blit(head, head_rect)
        self.window.blit(heart, heart_rect)
        self.window.blit(bullet, bullet_rect)

        health = font.render(f": {self.player_health}", False, "white")
        health_bounding_rect = health.get_rect(topleft=(40, 10))

        ammo_text = font.render(f": {self.ammo_count}", False, "white")
        ammo_bounding_rect = ammo_text.get_rect(topright=(WIDTH - 10, 10))

        kills_text = font.render(f": {int(self.kill_count)}", False, "white")
        kills_bounding_rect = kills_text.get_rect(midtop=(WIDTH / 2 + 14, 10))

        self.window.blit(health, health_bounding_rect)
        self.window.blit(ammo_text, ammo_bounding_rect)
        self.window.blit(kills_text, kills_bounding_rect)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return self._get_rgb()

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
