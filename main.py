from rl.env.deadly import WolfensteinDeadlyCorridorEnv
from rl.train import train_model
from rl.test import test_model

# Training

env = WolfensteinDeadlyCorridorEnv()
train_model(env=env, total_steps=5000000)

# Testing

# env = WolfensteinDefendTheCenterEnv(render_mode="human")
# MODEL_PATH = "./models/defend/model_4000000"
# test_model(env=env, model_path=MODEL_PATH, episodes=10)
