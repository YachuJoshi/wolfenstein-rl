import time
from stable_baselines3 import PPO


def test_random(env, episodes=10):
    for episode in range(episodes):
        _obs = env.reset()
        done = False
        reward = 0

        while True:
            env.render()
            action = env.action_space.sample()
            _obs, reward, done, _info = env.step(action)

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {reward}")

    env.close()


def test(env, model_path, episodes=10):
    model = PPO.load(model_path)

    for episode in range(1, episodes + 1):
        observation = env.reset()
        done = False
        reward = 0

        while True:
            env.render()
            action, _state = model.predict(observation)
            observation, reward, done, _info = env.step(action)

            if done:
                break

        time.sleep(0.5)
        print(f"Episode {episode}, Total Reward: {reward}")

    env.close()
