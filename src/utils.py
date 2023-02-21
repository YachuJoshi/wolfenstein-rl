from gym import Env
from typing import Tuple, Literal, Dict, Optional

Level = Literal["basic", "defend", "deadly"]
Mode = Literal["train", "test"]
DeadlyMode = Literal["easy", "medium", "hard", "insane"]

deadly_modes = {
    "1": "easy",
    "2": "medium",
    "3": "hard",
    "4": "insane",
}


def get_env(level: Level, mode: Mode, skill: Optional[DeadlyMode] = "easy") -> Env:
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    if level == "basic":
        from rl.env.basic import WolfensteinBasicEnv

        return WolfensteinBasicEnv(render_mode=mode)

    if level == "defend":
        from rl.env.defend import WolfensteinDefendTheCenterEnv

        return WolfensteinDefendTheCenterEnv(render_mode=mode)

    from rl.env.deadly import WolfensteinDeadlyCorridorEnv

    return WolfensteinDeadlyCorridorEnv(render_mode=mode, difficulty_mode=skill)


def get_dir(level: Level) -> Tuple[str, str]:
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./logs/{level}", f"./models/{level}/cnn"


def get_model_dir(level: Level, steps: int) -> str:
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./models/{level}/cnn/model_{steps}"


def get_n_steps(level: Level) -> int:
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    labels_map: Dict[str, int] = {
        "basic": 2048,
        "defend": 4096,
        "deadly": 8192,
    }

    return labels_map[level]


## Curriculum Learning - Deadly Corridor Utilities
def get_prev_model_path(prev_mode: DeadlyMode, steps: int = 700000) -> str:
    return f"./models/deadly/cnn/{prev_mode}/model_{steps}"


def get_deadly_model_dir(curr_mode: DeadlyMode) -> Tuple[str, str]:
    return f"./logs/deadly/{curr_mode}", f"./models/deadly/cnn/{curr_mode}"
