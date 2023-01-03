from rl.callback import TrainAndLoggingCallback
from stable_baselines3 import PPO

LOG_DIR = "./logs"
CHECKPOINT_DIR = "./models"


def train_model(env, total_steps, save_frequency=50000):
    callback = TrainAndLoggingCallback(
        check_freq=save_frequency, save_path=CHECKPOINT_DIR
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    model.learn(total_timesteps=total_steps, callback=callback)
