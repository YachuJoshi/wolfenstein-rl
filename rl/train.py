import torch
from gym import Env

from rl.core.ppo import PPO
from rl.common.callback import TrainAndLoggingCallback

from typing import Literal

Policy = Literal["MlpPolicy", "CnnPolicy"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train(
    env: Env,
    n_steps: int,
    policy: Policy,
    total_steps: int,
    log_dir: str,
    model_dir: str,
    save_frequency: int = 100000,
    **config
) -> None:
    device = get_device()
    callback = TrainAndLoggingCallback(check_freq=save_frequency, save_path=model_dir)
    model = PPO(
        policy=policy,
        env=env,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        n_steps=n_steps,
        **config
    )

    model.learn(total_timesteps=total_steps, callback=callback)


def curr_learn(
    new_env: Env,
    total_steps: int,
    model_save_path: str,
    prev_model_load_path: str,
    save_frequency: int = 100000,
) -> None:
    callback = TrainAndLoggingCallback(
        check_freq=save_frequency, save_path=model_save_path
    )
    model = PPO.load(prev_model_load_path)
    model.set_env(new_env)
    model.learn(total_timesteps=total_steps, callback=callback)
