from rl.env.defend import WolfensteinDefendTheCenterEnv
from rl.train import train_model
from rl.test import test_model

# Training

env = WolfensteinDefendTheCenterEnv()
train_model(env=env, total_steps=5000000)

# Testing

# env = WolfensteinDefendTheCenterEnv(render_mode="human")
# MODEL_PATH = "./models/defend/model_800000"
# test_model(env=env, model_path=MODEL_PATH, episodes=10)
