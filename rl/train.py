from rl.callback import TrainAndLoggingCallback
from stable_baselines3 import PPO

LOG_DIR = "./logs/defend"
CHECKPOINT_DIR = "./models/defend"


def train_model(env, total_steps, save_frequency=100000):
    callback = TrainAndLoggingCallback(
        check_freq=save_frequency, save_path=CHECKPOINT_DIR
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        # learning_rate=0.0001,
    )

    model.learn(total_timesteps=total_steps, callback=callback)
