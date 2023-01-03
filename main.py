from rl.env import WolfensteinEnv
from rl.train import train_model
from rl.test import test_model

# Training
env = WolfensteinEnv()
train_model(env=env, total_steps=400000)

# Testing
env = WolfensteinEnv(render_mode="human")
MODEL_PATH = "./models/model_400000"
test_model(env=env, model_path=MODEL_PATH)
