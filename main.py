import sys
from rl.train import train
from rl.test import test


def get_env(level, mode):
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    if level == "basic":
        from rl.env.basic import WolfensteinBasicEnv

        return WolfensteinBasicEnv(render_mode=mode)

    if level == "defend":
        from rl.env.defend import WolfensteinDefendTheCenterEnv

        return WolfensteinDefendTheCenterEnv(render_mode=mode)

    from rl.env.deadly import WolfensteinDeadlyCorridorEnv

    return WolfensteinDeadlyCorridorEnv(render_mode=mode)


def get_dir(level):
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./logs/{level}", f"./models/{level}"


def get_model_dir(level, steps):
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./models/{level}/model_{steps}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        level, mode = args[0].split("-")
        render_mode = None if mode == "train" else "human"
        env = get_env(level=level, mode=render_mode)
        log_dir, model_save_dir = get_dir(level=level)

        if mode == "train":
            train(
                env=env, total_steps=5000000, log_dir=log_dir, model_dir=model_save_dir
            )
        elif mode == "test":
            steps = sys.argv[2]
            model_load_dir = get_model_dir(level=level, steps=steps)
            test(env=env, model_path=model_load_dir)
        else:
            raise ValueError("Unknown mode")
