# Run Cart Pole env
import torch

import gymnasium as gym
import env
from sac.agent import SACAgent
from tqdm import tqdm
import numpy as np

environment = gym.make('CustomInvertedPendulum-v0')
print(environment.action_space.high[0])
agent = SACAgent(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0])

state, info = environment.reset(seed=42)
max_num_episodes = 1000
max_episode_length = 250
cost_arr = []
step_arr = []
steps_per_episode = []
total_steps = 0
longest_episode = 0

for _ in tqdm(range(max_num_episodes)):
    episode_cost = 0
    episode_steps = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state, reparameterize=False)
        next_state, cost, terminated, truncated, _ = environment.step(action)

        agent.remember((state, action, -cost, next_state, terminated))

        state = next_state

        episode_cost += cost
        episode_steps += 1
        total_steps += 1

        if terminated:
            break

    state, _ = environment.reset()
    
    for j in range(episode_steps):
        agent.train()
    
    if episode_steps > longest_episode:
        longest_episode = episode_steps

    steps_per_episode.append(episode_steps)
    cost_arr.append(episode_cost)
    step_arr.append(total_steps)

np.save("data/cartpole/arrays/sac-cost2-arr.npy", np.array(cost_arr))
np.save("data/cartpole/arrays/sac-step2-arr.npy", np.array(step_arr))
print(f"Longest Episode: {longest_episode}")
print(f"Average Steps per Episode: {np.mean(steps_per_episode)}")