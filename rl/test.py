from stable_baselines3 import PPO


def test_model(env, model_path, episodes=10):
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

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()
