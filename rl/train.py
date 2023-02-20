import torch
from gym import Env
from rl.ppo import PPO
from rl.utils.callback import TrainAndLoggingCallback


def get_device() -> torch.device:
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train(
    env: Env,
    total_steps: int,
    log_dir: str,
    model_dir: str,
    policy: str = "MlpPolicy",
    save_frequency: int = 100000,
) -> None:
    device = get_device()
    callback = TrainAndLoggingCallback(check_freq=save_frequency, save_path=model_dir)
    model = PPO(
        policy=policy,
        env=env,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        learning_rate=0.0001,
        n_steps=4096,
    )

    model.learn(total_timesteps=total_steps, callback=callback)
