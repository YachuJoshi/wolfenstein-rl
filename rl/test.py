import time
from gym import Env

from rl.core.ppo import PPO


def test_random(env: Env, episodes: int = 10) -> None:
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


def test(env: Env, model_path: str, episodes: int = 10) -> None:
    model = PPO.load(model_path)

    for episode in range(1, episodes + 1):
        observation = env.reset()
        done = False
        total_reward = 0

        while True:
            env.render()
            action, _state = model.predict(observation)
            observation, reward, done, _info = env.step(action)
            total_reward += reward

            if done:
                break

        time.sleep(0.5)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()
