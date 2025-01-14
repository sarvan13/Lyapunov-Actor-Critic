# Run Cart Pole env
import torch

import gymnasium as gym
import env
from sac.agent import SACAgent
from tqdm import tqdm
import numpy as np

environment = gym.make('InvertedPendulum-v5')

agent = SACAgent(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0])

state, info = environment.reset(seed=42)
max_num_episodes = 13000
max_episode_length = 250
num_gradient_updates = 1
cost_arr = []
step_arr = []
steps_per_episode = []
total_steps = 0
longest_episode = 0

for _ in tqdm(range(max_num_episodes)):
    episode_cost = 0
    episode_steps = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state)
        next_state, cost, terminated, truncated, _ = environment.step(action)
        cost = (next_state[0]/10) ** 2 + 20 * (next_state[1]/0.2) ** 2

        if np.abs(next_state[0]) >= 10 or np.abs(next_state[1]) >= 0.2:
            terminated = True
            cost = 100
        else:
            terminated = False

        agent.remember((state, action, -1*cost, next_state, terminated))

        episode_cost += cost
        episode_steps += 1
        total_steps += 1

        #environment.render()

        if terminated:
            break

    environment.reset()
    
    for j in range(num_gradient_updates):
        agent.train()
    
    if episode_steps > longest_episode:
        longest_episode = episode_steps

    steps_per_episode.append(episode_steps)
    cost_arr.append(episode_cost)
    step_arr.append(total_steps)

np.save("sac-cost-arr.npy", np.array(cost_arr))
np.save("sac-step-arr.npy", np.array(step_arr))
print(f"Longest Episode: {longest_episode}")
print(f"Average Steps per Episode: {np.mean(steps_per_episode)}")
