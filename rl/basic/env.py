import gym
import pygame
import numpy as np
from src.base import *
from src.screen import window, clock
from gym.spaces import Box, Discrete
from math import sin, cos, sqrt, atan2, degrees, pi
from src.textures import background, enemy, gun, textures, soldier_death


class WolfensteinBasicEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 120,
    }

    def __init__(self, render_mode=None):
        super(WolfensteinBasicEnv, self).__init__()

        shape = (2,)
        obs_low = np.ones(shape) * -np.inf
        obs_high = np.ones(shape) * np.inf

        # [ playerXPosition, enemyXPosition ]
        self.observation_space = Box(
            low=obs_low, high=obs_high, shape=(shape), dtype=np.float64
        )

        self.action_space = Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window, self.clock = window, clock
        self.enemies = []
        self.player_x = MAP_SCALE * 3 + 20.0
        self.player_y = MAP_SCALE * 8 + 20.0
        self.player_angle = pi
        self.enemy_dx = 1
        self.zbuffer = []

    def _get_obs(self):
        return np.array([self.player_x, self.enemies[0]["x"]])

    def _get_info(self):
        return {}

    def reset(self):
        self.soldier_death_count = 0
        self.enemy_dx = 1 if np.random.rand() > 0.5 else -1
        self.zbuffer = []
        self.reward = 0
        self.done = False
        self.player_x = MAP_SCALE * 3 + 20.0
        self.player_y = MAP_SCALE * 8 + 20.0
        self.player_angle = pi
        self.enemies = [
            {
                "image": enemy.subsurface(0, 0, 64, 64),
                "x": 160 if np.random.rand() > 0.5 else 300,
                "y": 360,
                "shift": 0.4,
                "scale": 1.0,
                "type": "soldier",
                "dead": False,
            },
        ]
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        self.done = False

        offset_x = sin(self.player_angle) * MAP_SPEED
        offset_y = cos(self.player_angle) * MAP_SPEED
        distance_thresh_x = 20 if offset_x > 0 else -20
        distance_thresh_y = 20 if offset_y > 0 else -20

        target_x = int(self.player_y / MAP_SCALE) * MAP_SIZE + int(
            (self.player_x + offset_x + distance_thresh_x) / MAP_SCALE
        )
        target_y = int(
            (self.player_y + offset_y + distance_thresh_y) / MAP_SCALE
        ) * MAP_SIZE + int(self.player_x / MAP_SCALE)

        # 0 -> Strafe Left
        # 1 -> Strafe Right
        # 2 -> Attack

        if action == 0:
            target_x = int(self.player_y / MAP_SCALE) * MAP_SIZE + int(
                (self.player_x - offset_x - distance_thresh_x) / MAP_SCALE
            )
            if MAP_BASIC[target_x] in " e":
                self.player_x += offset_y

            if MAP_BASIC[target_y] in " e":
                self.player_y -= offset_x

        elif action == 1:
            target_x = int(self.player_y / MAP_SCALE) * MAP_SIZE + int(
                (self.player_x + offset_x + distance_thresh_x) / MAP_SCALE
            )

            if MAP_BASIC[target_x] in " e":
                self.player_x -= offset_y
            if MAP_BASIC[target_y] in " e":
                self.player_y += offset_x

        elif action == 2:
            if gun["animation"] == False:
                gun["animation"] = True

            if not self.enemies[0]["dead"]:
                self.reward -= 0.1

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
                if target_square not in range(len(MAP_BASIC)):
                    break
                if MAP_BASIC[target_square] not in " e":
                    texture_y = (
                        MAP_BASIC[target_square]
                        if MAP_BASIC[target_square] != "T"
                        else "I"
                    )
                    if MAP_BASIC[target_square] == "E":
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
                if target_square not in range(len(MAP_BASIC)):
                    break
                if MAP_BASIC[target_square] not in " e":
                    texture_x = (
                        MAP_BASIC[target_square]
                        if MAP_BASIC[target_square] != "O"
                        else "J"
                    )
                    if MAP_BASIC[target_square] == "E":
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

        # position & scale sprites
        for sprite in self.enemies:
            sprite_x = sprite["x"] - self.player_x
            sprite_y = sprite["y"] - self.player_y
            sprite_distance = sqrt(sprite_x * sprite_x + sprite_y * sprite_y)
            sprite2player_angle = atan2(sprite_x, sprite_y)
            player2sprite_angle = sprite2player_angle - self.player_angle

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
                        self.enemy_dx *= -1

                    sprite["x"] += self.enemy_dx

                    if (
                        abs(shift_rays) < 20
                        and sprite_distance < 500
                        and gun["animation"]
                    ):
                        sprite["image"] = soldier_death[
                            int(self.soldier_death_count / 8)
                        ]
                        self.soldier_death_count += 1
                        if self.soldier_death_count >= 16:
                            sprite["dead"] = True
                            self.soldier_death_count = 0
                            self.enemy_dx = 0
                            self.reward += 10
                            self.done = True

                else:
                    sprite["image"] = soldier_death[-1]
                    self.enemy_dx = 0
                    self.done = True
                if gun["shot_count"] > 16 and sprite["image"] in [
                    soldier_death[0],
                    soldier_death[1],
                    soldier_death[2],
                ]:
                    try:
                        sprite["image"] = soldier_death[
                            int(self.soldier_death_count / 8) + 2
                        ]
                        self.enemy_dx = 0
                        self.reward += 10
                        self.done = True

                    except:
                        pass
                    self.soldier_death_count += 1

                    if self.soldier_death_count >= 32:
                        sprite["dead"] = True
                        self.soldier_death_count = 0
                        self.reward += 10
                        self.done = True

                if not sprite["dead"] and sprite_distance <= 10:
                    self.player_x -= offset_x
                    self.player_y -= offset_y

            sprite_image = pygame.transform.scale(
                sprite["image"], (int(sprite_height), int(sprite_height))
            )

            self.zbuffer.append(
                {
                    "image": sprite_image,
                    "x": sprite_ray - int(sprite_height / 2),
                    "y": 100 - sprite_height * sprite["shift"],
                    "distance": sprite_distance,
                }
            )

        if not self.done:
            self.reward -= 1

        observation = self._get_obs()
        reward = self.reward
        done = self.done
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        self.window.blit(background, (0, 0))

        self.zbuffer = sorted(self.zbuffer, key=lambda k: k["distance"], reverse=True)
        for item in self.zbuffer:
            window.blit(item["image"], (item["x"], item["y"]))

        # render gun / gun animation
        self.window.blit(gun["default"], (60, 20))
        if gun["animation"]:
            gun["animation"] = True
            window.blit(gun["shot"][int(gun["shot_count"] / 5)], (60, 20))
            gun["shot_count"] += 1
            if gun["shot_count"] >= 20:
                gun["shot_count"] = 0
                gun["animation"] = False
            pygame.display.flip()

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
