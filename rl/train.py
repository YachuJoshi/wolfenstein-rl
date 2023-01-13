from rl.callback import TrainAndLoggingCallback
from stable_baselines3 import PPO


def train(
    env,
    total_steps,
    log_dir,
    model_dir,
    save_frequency=100000,
):
    callback = TrainAndLoggingCallback(check_freq=save_frequency, save_path=model_dir)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0002,
    )

    model.learn(total_timesteps=total_steps, callback=callback)
