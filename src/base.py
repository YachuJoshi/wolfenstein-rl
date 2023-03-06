from math import pi
from typing import Tuple, List

# screen
WIDTH: int = 320
HEIGHT: int = 200

# camera
FOV: float = pi / 3
HALF_FOV: float = FOV / 2
STEP_ANGLE: float = FOV / WIDTH
CENTRAL_RAY = int(WIDTH / 2) - 1
DOUBLE_PI: int = 2 * pi

# map
# MAP_SIZE = 22  # -> BASIC
# MAP_SIZE = 10  # -> DEFEND
# MAP_SIZE = 22  # -> DEADLY CORRIDOR
MAP_SCALE: int = 64

MAP_DEMO = list(
    "SSSSSSSSSSSSSSSSSSSSSS"
    "S     SS             S"
    "S          SSSSSS S  S"
    "S     SS       SS S  S"
    "SSSSSSSSSSSS   SS SSSS"
    "S      SS      SS    S"
    "S      SS      SS    S"
    "S              SSS   S"
    "S         SS     S   S"
    "SSSSSSS   SS     SSSSS"
    "S         SSSSS  SS  S"
    "S                SS  S"
    "SSSSSSSSSSSSSSSSSSS  S"
    "SSSSSSSSSSSSSSSSSSS  S"
    "S        S        S  S"
    "S  SSSSSSSSSSSSSSSS  S"
    "S  SSSSSSSSSSS   SS  S"
    "S  S        SS       S"
    "S  S        SS       S"
    "S           SS       S"
    "S  S        SS       S"
    "SSSSSSSSSSSSSSSSSSSSSS"
)

# "SSTETSSSWWWW   WS SSSS" -> Door

MAP_BASIC = list(
    "SSSSSSSSSSSSSSSSSSSSSS"
    "S                    S"
    "S                    S"
    "S                    S"
    "SSSSSSS              S"
    "S     S              S"
    "S     S              S"
    "S     S              S"
    "S     S              S"
    "SSSSSSS              S"
    "S                    S"
    "SSSSSSSSSSSSSSSSSSSSSS"
)


MAP_DEFEND = list(
    "BBBBBBBBBB"
    "B        B"
    "B        B"
    "B        B"
    "B        B"
    "B        B"
    "B        B"
    "B        B"
    "B        B"
    "BBBBBBBBBB"
)

MAP_DEADLY_CORRIDOR = list(
    "SSSSSSSSSSSSSSSSSSSSSS"
    "S      S      S      S"
    "S                    S"
    "S      S      S      S"
    "SSSSSSSSSSSSSSSSSSSSSS"
)

MAP_LIST = [
    {
        "name": "DEMO",
        "map": MAP_DEMO,
        "map_size": 22,
    },
    {
        "name": "BASIC",
        "map": MAP_BASIC,
        "map_size": 22,
    },
    {
        "name": "DEFEND",
        "map": MAP_DEFEND,
        "map_size": 10,
    },
    {
        "name": "DEADLY",
        "map": MAP_DEADLY_CORRIDOR,
        "map_size": 22,
    },
]


def get_map_details(
    map_name: str,
) -> Tuple[List[str], int, float, float]:
    map = list(filter(lambda map_item: map_item["name"] == map_name, MAP_LIST))[0]
    MAP_RANGE: float = map["map_size"] * MAP_SCALE
    MAP_SPEED: float = (MAP_SCALE / 2) / 10
    return (map["map"], map["map_size"], MAP_RANGE, MAP_SPEED)
