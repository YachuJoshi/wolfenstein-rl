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
    """Get corresponding gym env for training

    Args:
        level (Level): Basic | Defend | Deadly
        mode (Mode): Train | Test
        skill (Optional[DeadlyMode], optional): For Deadly Mode | Specify 'easy', 'medium', 'hard'. Defaults to "easy".

    Raises:
        ValueError: Need a valid level

    Returns:
        Env: Gym Environment
    """
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
    """Get model log and save directory

    Args:
        level (Level): Basic | Defend | Deadly

    Raises:
        ValueError: Need a valid level

    Returns:
        Tuple[str, str]: ( Tensorboard log directory, Model save directory )
    """
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./logs/{level}", f"./models/{level}/cnn"


def get_model_dir(level: Level, steps: int) -> str:
    """Get corresponding level's model directory

    Args:
        level (Level): Basic | Defend | Deadly
        steps (int): Model steps

    Raises:
        ValueError: Need a valid level

    Returns:
        str: Model load path
    """
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./models/{level}/cnn/model_{steps}"


def get_n_steps(level: Level) -> int:
    """Get no. of rollout steps for each level

    Args:
        level (Level): Basic | Defend | Deadly

    Raises:
        ValueError: Need a valid level

    Returns:
        int: No. of rollout steps
    """
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    labels_map: Dict[str, int] = {
        "basic": 2048,
        "defend": 4096,
        "deadly": 8192,
    }

    return labels_map[level]


## Curriculum Learning - Deadly Corridor Utilities
def get_deadly_model_path(deadly_mode: DeadlyMode, steps: int = 800000) -> str:
    """Get deadly model load path for curriculum learning.

    Args:
        deadly_mode (DeadlyMode): easy | medium | hard
        steps (int, optional): Specifies the model steps to load. Defaults to 800000.

    Returns:
        str: Deadly model load path
    """
    return f"./models/deadly/cnn/{deadly_mode}/model_{steps}"


def get_deadly_model_dir(curr_mode: DeadlyMode) -> Tuple[str, str]:
    """Get model log and save directory

    Args:
        curr_mode (DeadlyMode): easy | medium | hard

    Returns:
        Tuple[str, str]:  ( Tensorboard log directory, Model save directory )
    """
    return f"./logs/deadly/{curr_mode}", f"./models/deadly/cnn/{curr_mode}"
