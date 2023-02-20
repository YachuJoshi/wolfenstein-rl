import torch
from rl.ppo import PPO
from rl.utils.callback import TrainAndLoggingCallback


def get_device():
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train(
    env,
    total_steps,
    log_dir,
    model_dir,
    policy="MlpPolicy",
    save_frequency=100000,
):
    device = get_device()
    callback = TrainAndLoggingCallback(check_freq=save_frequency, save_path=model_dir)
    # model = PPO(
    #     policy=policy,
    #     env=env,
    #     verbose=1,
    #     device=device,
    #     tensorboard_log=log_dir,
    #     learning_rate=0.0001,
    #     n_steps=4096,  # 8192 for deadly,4096 for defend,2048 for basic
    # )
    model = PPO(
        policy=policy,
        env=env,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        learning_rate=0.00001,
        n_steps=8192,  # 8192 for deadly,4096 for defend,2048 for basic
        clip_range=0.1,
        gamma=0.95,
        gae_lambda=0.9,
    )  #### this one is for deadly corridor level

    model.learn(total_timesteps=total_steps, callback=callback)
