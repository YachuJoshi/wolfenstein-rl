from env import WolfensteinEnv
from callback import TrainAndLoggingCallback
from stable_baselines3 import PPO

LOG_DIR = "./logs"
CHECKPOINT_DIR = "./models"

# env = WolfensteinEnv()
env = WolfensteinEnv(render_mode="human")
# episodes = 5

# for episode in range(1, episodes + 1):
#     obs = env.reset()
#     total_reward = 0

#     while True:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         total_reward += reward

#         if done:
#             break

#     print(f"Episode: {episode} Reward: {reward}")


# callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR)

# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log=LOG_DIR,
# )

# model.learn(total_timesteps=400000, callback=callback)

model = PPO.load("./models/model_400000")
episodes = 10

for episode in range(1, episodes + 1):
    observation = env.reset()
    done = False
    total_reward = 0

    while True:
        env.render()
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
