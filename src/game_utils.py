import pygame
from math import sin, cos
from src.base import MAP, MAP_SIZE, MAP_SCALE


def draw_minimap(
    window: pygame.Surface,
    sprites,
    player_x: float,
    player_y: float,
    player_angle: float,
):
    for row in range(MAP_SIZE):
        for col in range(10):
            pygame.draw.rect(
                window,
                (100, 100, 100) if MAP[row * 10 + col] != " " else (200, 200, 200),
                (col * 5, row * 5, 5, 5),
            )
        pygame.draw.circle(
            window,
            (255, 0, 0),
            (int((player_x / MAP_SCALE) * 5), int((player_y / MAP_SCALE) * 5)),
            2,
        )
        pygame.draw.line(
            window,
            (255, 0, 0),
            ((player_x / MAP_SCALE) * 5, (player_y / MAP_SCALE) * 5),
            (
                (player_x / MAP_SCALE) * 5 + sin(player_angle) * 5,
                (player_y / MAP_SCALE) * 5 + cos(player_angle) * 5,
            ),
            1,
        )
        for sprite in sprites:
            if sprite["type"] == "soldier" and not sprite["dead"]:
                pygame.draw.circle(
                    window,
                    (0, 0, 255),
                    (
                        int((sprite["x"] / MAP_SCALE) * 5),
                        int((sprite["y"] / MAP_SCALE) * 5),
                    ),
                    2,
                )


def show_fps(window: pygame.Surface, clock: pygame.time.Clock):
    # print FPS to screen
    font = pygame.font.SysFont("Monospace Regular", 30)
    fps_surface = font.render("FPS: " + str(int(clock.get_fps())), False, (255, 0, 0))
    window.blit(fps_surface, (120, 0))
