import sys

from rl.config import *
from rl.test import test, test
from rl.train import train, curr_learn
from rl.env.deadly_basic import WolfensteinDeadlyCorridorEnv

from src.utils import *
from src.argsparser import args


if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_mode = None if args.train else "human"
        n_steps = get_n_steps(args.level)
        config = DEADLY_CONFIG if args.level == "deadly" else NORMAL_CONFIG

        if args.level in ("basic", "defend"):
            env = get_env(level=args.level, mode=render_mode)
            log_dir, model_save_dir = get_dir(level=args.level)

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
                model_load_path = get_model_dir(level=args.level, steps=args.steps)
                test(env=env, model_path=model_load_path)

            else:
                raise ValueError("Unknown mode")

        else:
            if args.train:
                skill_level = deadly_modes[str(args.skill)]

                if args.curr:
                    prev_skill_level = deadly_modes[str(args.skill - 1)]
                    prev_model_path = get_deadly_model_path(prev_skill_level)
                    new_log_dir, new_model_save_path = get_deadly_model_dir(
                        curr_mode=skill_level
                    )
                    new_env = get_env(
                        level=args.level,
                        mode=render_mode,
                        skill=skill_level,
                    )
                    curr_learn(
                        new_env=new_env,
                        total_steps=2000000,
                        model_save_path=new_model_save_path,
                        prev_model_load_path=prev_model_path,
                    )
                else:
                    # env = get_env(level=args.level, mode=render_mode, skill=skill_level)
                    # log_dir, model_save_dir = get_deadly_model_dir(
                    #     curr_mode=skill_level
                    # )
                    env = WolfensteinDeadlyCorridorEnv(
                        render_mode=None, difficulty_mode="medium"
                    )
                    log_dir, model_save_dir = get_dir(level=args.level)
                    train(
                        env=env,
                        n_steps=n_steps,
                        policy="CnnPolicy",
                        total_steps=2000000,
                        log_dir=log_dir,
                        model_dir=model_save_dir,
                        **config
                    )
            else:
                enemy_skill = deadly_modes[str(args.skill)]
                # env = get_env(level=args.level, mode=render_mode, skill=enemy_skill)
                # model_load_path = get_deadly_model_path(enemy_skill, steps=args.steps)
                env = WolfensteinDeadlyCorridorEnv(
                    render_mode="human", difficulty_mode="medium"
                )
                model_load_path = get_model_dir(level=args.level, steps=args.steps)
                test(env=env, model_path=model_load_path)
