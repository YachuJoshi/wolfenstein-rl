from rl.defend_env import WolfensteinDefendTheCenterEnv
from rl.train import train_model
from rl.test import test_model

# Training
env = WolfensteinDefendTheCenterEnv(render_mode="human")
train_model(env=env, total_steps=800000)


# Testing
# env = WolfensteinEnv(render_mode="human")
# MODEL_PATH = "./models/basic/model_600000"
# test_model(env=env, model_path=MODEL_PATH, episodes=20)

# Random Action
# env = WolfensteinDefendTheCenterEnv(render_mode="human")
# for episode in range(10):
#     obs = env.reset()
#     done = False
#     total_reward = 0

#     while True:
#         env.render()
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         total_reward += reward

#         if done:
#             break

#     print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
