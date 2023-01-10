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
MAP_SIZE = 10  # -> DEFEND
MAP_SCALE = 64
MAP_RANGE = MAP_SIZE * MAP_SCALE
MAP_SPEED = (MAP_SCALE / 2) / 10
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
    "SSTSTSS              S"
    "S     S              S"
    "S     S              S"
    "S     O              S"
    "S     S              S"
    "SSSSSSS              S"
    "S                    S"
    "SSSSSSSSSSSSSSSSSSSSSS"
)


MAP = MAP_DEFEND = list(
    "DDDDDDDDDD"
    "B        E"
    "B        E"
    "B        E"
    "B        E"
    "B        E"
    "B        E"
    "B        E"
    "B        E"
    "SSSSSSSSSS"
)
