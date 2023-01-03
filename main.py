from rl.defend_env import WolfensteinEnv
from rl.train import train_model
from rl.test import test_model

from stable_baselines3.common.env_checker import check_env

env = WolfensteinEnv()
check_env(env)

# Training
# env = WolfensteinEnv()
# train_model(env=env, total_steps=800000)


# Testing
# env = WolfensteinEnv(render_mode="human")
# MODEL_PATH = "./models/basic/model_600000"
# test_model(env=env, model_path=MODEL_PATH, episodes=20)
