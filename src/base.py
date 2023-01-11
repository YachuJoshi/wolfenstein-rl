from math import pi

# screen
WIDTH = 320
HEIGHT = 200

# camera
FOV = pi / 3
HALF_FOV = FOV / 2
STEP_ANGLE = FOV / WIDTH
CENTRAL_RAY = int(WIDTH / 2) - 1
DOUBLE_PI = 2 * pi

# map
# MAP_SIZE = 22  # -> BASIC
# MAP_SIZE = 10  # -> DEFEND
# MAP_SIZE = 22  # -> DEADLY CORRIDOR
MAP_SCALE = 64

# MAP = list(
#     "SSSSSSSBBSBBBBBBBBBBBB"
#     "S     SB             B"
#     "S     SB   BBBBBB B  B"
#     "S     SB       WB B  F"
#     "SSTETSSSWWWW   WB BBBB"
#     "S      SW      WB    B"
#     "S     SSW      WB    B"
#     "S     OBBBBB   WWB   B"
#     "S     E   BB     B   B"
#     "SSSSSSOB  SB     BTETS"
#     "S     SB  BBBBB  BS  S"
#     "S     SB         BS  S"
#     "SSSSSSSBBBBBBBBBBBS  S"
#     "DDDDDDDDDOSSSSSSSSO  S"
#     "D        E        E  S"
#     "D  DDDDDDOSSSSSSSSO  M"
#     "D  DDXDXDXDDDS   SS  S"
#     "D  D        DS       S"
#     "D  D        SS       S"
#     "D           SS       M"
#     "D  D        DS       S"
#     "DDDDDXDXDXDDDDSSSSSSSS"
# )

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
    "SSSSSSSSSS"
    "S        S"
    "S        S"
    "S        S"
    "S        S"
    "S        S"
    "S        S"
    "S        S"
    "S        S"
    "SSSSSSSSSS"
)

MAP_DEADLY_CORRIDOR = list(
    "SSSSSSSSSSSSSSSSSSSSSS"
    "S     SS       SS    S"
    "S                    S"
    "S         SS     S   S"
    "SSSSSSSSSSSSSSSSSSSSSS"
)

MAP_LIST = [
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


def get_map_details(map_name: str):
    map = list(filter(lambda map_item: map_item["name"] == map_name, MAP_LIST))[0]
    MAP_RANGE: float = map["map_size"] * MAP_SCALE
    MAP_SPEED: float = (MAP_SCALE / 2) / 10
    return (map["map"], map["map_size"], MAP_RANGE, MAP_SPEED)
