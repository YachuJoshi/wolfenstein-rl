import torch
from gym import Env

from rl.core.ppo import PPO
from rl.common.callback import TrainAndLoggingCallback

from typing import Literal

Policy = Literal["MlpPolicy", "CnnPolicy"]


def get_device() -> torch.device:
    """Get Device

    Returns:
        torch.device: Device to perform training / operation on
    """
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
    """

    Args:
        env (Env): Gym Environment to train the model on
        n_steps (int): No. of rollout steps
        policy (Policy): ANN ( MLP ) Policy / CNN Policy
        total_steps (int): Total no. of timesteps
        log_dir (str): Tensorboard log save directory
        model_dir (str): Location to save the model
        save_frequency (int, optional): Specifies the timestep for how frequently the model should save.. Defaults to 100000.
    """
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
    tensorboard_log: str,
    prev_model_load_path: str,
    save_frequency: int = 100000,
) -> None:
    """
    Initiate Curriculum Learning

    Args:
        new_env (Env): The new gym environment for training > The new gym env should have similar observation and action space
        total_steps (int): The total timesteps for training the model
        model_save_path (str): Location to save the model
        tensorboard_log (str): Tensorboard log save directory
        prev_model_load_path (str): Location of previous model to load
        save_frequency (int, optional): Timestep for how frequently the model should save. Defaults to 100000.
    """
    callback = TrainAndLoggingCallback(
        check_freq=save_frequency, save_path=model_save_path
    )
    model = PPO.load(prev_model_load_path)
    model.tensorboard_log = tensorboard_log
    model.set_env(new_env)
    model.learn(total_timesteps=total_steps, callback=callback)
