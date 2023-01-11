from rl.deadly_corridor.env import WolfensteinDeadlyCorridorEnv
from rl.train import train_model
from rl.test import test_model

# Training

# env = WolfensteinDeadlyCorridorEnv()
# train_model(env=env, total_steps=5000000)

# Testing

# env = WolfensteinDefendTheCenterEnv(render_mode="human")
# MODEL_PATH = "./models/defend/best_model_2000000"
# test_model(env=env, model_path=MODEL_PATH, episodes=10)
