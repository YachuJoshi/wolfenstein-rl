import torch
from gym import Env

from rl.ppo import PPO
from rl.utils.callback import TrainAndLoggingCallback

from typing import Literal

Policy = Literal["MlpPolicy", "CnnPolicy"]


def get_device() -> torch.device:
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")

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
    env: Env,
    new_env: Env,
    n_steps: int,
    policy: Policy,
    total_steps: int,
    log_dir: str = "./logs/deadly",
    model_save_path: str = "./models/deadly/cnn",
    prev_model_load_path: str = "./models/deadly/cnn/easy/model_700000",
    save_frequency: int = 100000,
    **config
) -> None:
    device = get_device()
    callback = TrainAndLoggingCallback(
        check_freq=save_frequency, save_path=model_save_path
    )
    model = PPO(
        policy=policy,
        env=env,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        n_steps=n_steps,
        **config
    )
    model.load(prev_model_load_path)
    model.set_env(new_env)

    model.learn(total_timesteps=total_steps, callback=callback)
