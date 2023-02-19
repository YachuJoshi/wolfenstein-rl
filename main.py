import sys
from rl.train import train
from rl.test import test, test
from src.argsparser import args
from src.utils import get_env, get_dir, get_model_dir

if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_mode = None if args.train else "human"
        env = get_env(level=args.level, mode=render_mode)
        log_dir, model_save_dir = get_dir(level=args.level)

        if args.train:
            train(
                env=env,
                policy="CnnPolicy",
                total_steps=5000000,
                log_dir=log_dir,
                model_dir=model_save_dir,
            )
        elif args.test:
            model_load_dir = get_model_dir(level=args.level, steps=args.steps)
            test(env=env, model_path=model_load_dir)
        else:
            raise ValueError("Unknown mode")
