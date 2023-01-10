from rl.defend_env import WolfensteinDefendTheCenterEnv
from rl.train import train_model
from rl.test import test_model

# Training
env = WolfensteinDefendTheCenterEnv()
train_model(env=env, total_steps=5000000)

# Testing
# env = WolfensteinDefendTheCenterEnv(render_mode="human")
# MODEL_PATH = "./models/defend/best_model_2000000"
# test_model(env=env, model_path=MODEL_PATH, episodes=10)

# episodes = 10
# for episode in range(episodes):
#     done = False
#     obs = env.reset()
#     reward = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         obs, reward, done, _ = env.step(action)

#     print(f"Episode: {episode}, Reward: {reward}")
