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

from collections import namedtuple
from typing import Union, Tuple, Literal, Dict

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
INITIAL_ANGLE = 1.55
GEM_POSITION = {
    "x": 1280.0,
    "y": 160.0,
}

TypeStep = Tuple[np.ndarray, float, bool, dict]
DifficultyMode = Literal["easy", "medium", "hard", "insane"]
RenderMode = Union[Literal["human"], Literal["rgb_array"], None]

MODE: Dict[str, float] = {
    "easy": 0.8,
    "medium": 0.6,
    "hard": 0.4,
    "insane": 0.2,
}

MAP, MAP_SIZE, MAP_RANGE, MAP_SPEED = get_map_details("DEADLY")
MAX_STEPS = 4200


class WolfensteinDeadlyCorridorEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 120,
    }

    def __init__(
        self,
        render_mode: RenderMode = None,
        difficulty_mode: DifficultyMode = "easy",
    ) -> None:
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
        self.player_health = 100
        self.steps = 0
        self.zbuffer = []

        # Reward Shaping
        self.hitcount = 0
        self.ammo_count = 50
        self.damage_taken = 0
        self.enemy_death_count = 0
        self.ammo = self.ammo_count

        # Curriculum Learning
        self.mode = difficulty_mode
        self.threshold = MODE[difficulty_mode]

    def _get_obs(self) -> np.ndarray:
        return self._render_frame()

    def _get_info(self) -> Dict[str, int]:
        return {"ammo": self.ammo_count}

    def _enemy_hit(self, enemy: Enemy) -> None:
        enemy.dead = True
        enemy.dx = 0
        self.enemy_death_count += 1

    def _transform_image(self, observation: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        reshaped = np.reshape(resized, (100, 160, 1))
        return reshaped

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.reward = 0
        self.zbuffer = []
        self.done = False
        self.player_x = 86.0
        self.player_y = 160.0
        self.hitcount = 0
        self.ammo_count = 50
        self.damage_taken = 0
        self.enemy_death_count = 0
        self.ammo = self.ammo_count
        self.player_health = 100
        self.player_angle = INITIAL_ANGLE
        # self.enemies = [
        #     Enemy(id=1, x=304.0, y=98.0),
        #     Enemy(id=2, x=450.0, y=236.0),
        #     Enemy(id=3, x=680.0, y=98.0, distance_threshold=180),
        #     Enemy(id=4, x=850.0, y=236.0, distance_threshold=180),
        #     Enemy(id=5, x=1200.0, y=98.0, distance_threshold=180),
        #     Enemy(id=6, x=1250.0, y=236.0, distance_threshold=180),
        # ]
        self.enemies = [
            Enemy(id=1, x=384.0, y=98.0),
            Enemy(id=2, x=384.0, y=236.0),
            Enemy(id=3, x=740.0, y=98.0),
            Enemy(id=4, x=740.0, y=236.0),
            Enemy(id=5, x=1200.0, y=98.0),
            Enemy(id=6, x=1200.0, y=236.0),
        ]

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action: int) -> TypeStep:
        self.reward = 0
        self.steps += 1
        self.done = False
        rewardX = 0

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
            self.player_angle += 0.05

        elif action == 1:
            self.player_angle -= 0.05

        elif action == 2 and self.player_x < 1265.0:
            is_moving = True
            if MAP[target_x] in " e":
                self.player_x += offset_x

            if (
                not (self.enemies[0].dead and self.enemies[1].dead)
                and self.player_x > 87
            ):
                self.player_x = 87
                is_moving = False
            if (
                not (self.enemies[2].dead and self.enemies[3].dead)
                and self.player_x > 460
            ):
                self.player_x = 460
                is_moving = False
            if (
                not (self.enemies[4].dead and self.enemies[5].dead)
                and self.player_x > 910
            ):
                self.player_x = 910
                is_moving = False

            rewardX = offset_x if is_moving else 0

        elif action == 3 and self.player_x > 86.0:
            if MAP[target_x] in " e":
                self.player_x -= offset_x
                rewardX = -offset_x

        elif action == 4:
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
                if distance_x < enemy.distance_threshold + 50:
                    enemy.is_attacking = True
                else:
                    enemy.is_attacking = False

                if enemy.is_attacking:
                    enemy.image = enemy.attack_animation_list[
                        int(enemy.attack_index / 8)
                    ]
                    enemy.attack_index += 0.5

                    if enemy.attack_index > 15:
                        enemy.attack_index = 0

                    # Getting Hit Probability
                    if np.random.rand() > self.threshold:
                        self.player_health -= 1

                # Shoot & Enemy Dead
                if (
                    abs(shift_rays) < 20
                    and distance_x < enemy.distance_threshold
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
            if gun["shot_count"] > 4 and enemy.image in [
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

        self.reward += rewardX

        if self.player_health <= 0:
            self.reward -= 100
            self.done = True

        if self.player_x >= GEM_POSITION["x"] - 20:
            self.done = True

        current_ammo_count = self.ammo_count
        current_damage_taken = 100 - self.player_health
        current_enemy_death_count = self.enemy_death_count

        # Difference -> current - previous
        damage_difference = abs(
            current_damage_taken - self.damage_taken
        )  # Negative Reinforcement
        self.damage_taken = current_damage_taken

        enemy_count_difference = abs(
            current_enemy_death_count - self.hitcount
        )  # Positive Reinforcement
        self.hitcount = current_enemy_death_count

        ammo_difference = abs(current_ammo_count - self.ammo)  # Negative Reinforcement
        self.ammo = current_ammo_count

        if self.steps > MAX_STEPS:
            self.done = True

        observation = self._get_obs()
        reward = (
            self.reward
            + damage_difference * -20
            + enemy_count_difference * 100
            + ammo_difference * -5
        )
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

        kills_text = font.render(f": {int(self.enemy_death_count)}", False, "white")
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

    def __str__(self) -> str:
        return f"WolfensteinDeadlyCorridorEnv: Difficulty: {self.mode}"

    def __repr__(self) -> str:
        return f"WolfensteinDeadlyCorridorEnv: Difficulty: {self.mode}"
