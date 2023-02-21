import sys

from rl.config import *
from rl.test import test, test
from rl.train import train, curr_learn

from src.utils import *
from src.argsparser import args


if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_mode = None if args.train else "human"
        env = get_env(level=args.level, mode=render_mode)
        log_dir, model_save_dir = get_dir(level=args.level)
        n_steps = get_n_steps(args.level)
        config = DEADLY_CONFIG if args.level == "deadly" else NORMAL_CONFIG

        if args.train:
            if args.curr:
                prev_skill_level = deadly_modes[str(args.skill - 1)]
                current_skill_level = deadly_modes[str(args.skill)]
                prev_model_path = get_prev_model_path(prev_skill_level)
                new_log_dir, new_model_save_path = get_deadly_model_dir(
                    curr_mode=current_skill_level
                )
                prev_env = get_env(
                    level=args.level,
                    mode=None,
                    skill=prev_skill_level,
                )
                new_env = get_env(
                    level=args.level,
                    mode=render_mode,
                    skill=current_skill_level,
                )
                curr_learn(
                    env=prev_env,
                    new_env=new_env,
                    n_steps=n_steps,
                    policy="CnnPolicy",
                    total_steps=2000000,
                    log_dir=new_log_dir,
                    model_save_path=new_model_save_path,
                    prev_model_load_path=prev_model_path,
                    **config
                )
            else:
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
