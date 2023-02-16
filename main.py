import sys
import argparse
from rl.train import train
from rl.test import test

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="N",
    help="Random Seed (Default: 42)",
)
parser.add_argument(
    "--level",
    type=str,
    metavar="basic | defend | deadly",
    help="The Level For The Game",
)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--train", action="store_true", help="Initiate Training Mode")
mode.add_argument("--test", action="store_true", help="Initiate Testing Mode")
parser.add_argument(
    "--steps",
    type=int,
    help="Specify a model to be loaded based on time-step",
    required="--test" in sys.argv,
)
args = parser.parse_args()


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
        render_mode = None if args.train else "human"
        env = get_env(level=args.level, mode=render_mode)
        log_dir, model_save_dir = get_dir(level=args.level)

        if args.train:
            train(
                env=env, total_steps=5000000, log_dir=log_dir, model_dir=model_save_dir
            )
        elif args.test:
            model_load_dir = get_model_dir(level=args.level, steps=args.steps)
            test(env=env, model_path=model_load_dir)
        else:
            raise ValueError("Unknown mode")
