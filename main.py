# Run Cart Pole env

import gymnasium as gym
import env

environment = gym.make("CartPoleadv-v1", render_mode="human")

observation, info = environment.reset(seed=42)
for _ in range(10000):
    action = environment.action_space.sample()
    observation, reward, terminated, truncated = environment.step(action)

    print(str((observation,reward)))

    environment.render()
                                                
    if terminated:
        observation, info = environment.reset()
        print(terminated)
        print(truncated)
environment.close()

# import gymnasium as gym
# import env

# # Ensure this matches the `id` in your registration
# env = gym.make("CartPoleadv-v1")  
# obs, _ = env.reset()
# print("Environment initialized successfully!")

