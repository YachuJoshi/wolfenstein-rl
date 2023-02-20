import sys

from rl.config import *
from rl.train import train
from rl.test import test, test

from src.utils import *
from src.argsparser import args


if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_mode = "human" if args.train else "human"
        env = get_env(level=args.level, mode=render_mode)
        log_dir, model_save_dir = get_dir(level=args.level)
        n_steps = get_n_steps(args.level)
        config = DEADLY_CONFIG if args.level == "deadly" else NORMAL_CONFIG

        if args.train:
            train(
                env=env,
                n_steps=n_steps,
                policy="CnnPolicy",
                total_steps=2000000,
                log_dir=log_dir,
                model_dir=model_save_dir,
                **config
            )
        elif args.test:
            model_load_dir = get_model_dir(level=args.level, steps=args.steps)
            test(env=env, model_path=model_load_dir)
        else:
            raise ValueError("Unknown mode")
