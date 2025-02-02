# Run Cart Pole env
import torch

import gymnasium as gym
import env
from lac.agent import LAC
from tqdm import tqdm
import numpy as np

environment = gym.make('CustomInvertedPendulum-v0')
print(environment.action_space.high[0])
agent = LAC(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0])

state, info = environment.reset(seed=42)
max_num_episodes = 1000
max_episode_length = 250
cost_arr = []
step_arr = []
steps_per_episode = []
lambda_arr = []
beta_arr = []
total_steps = 0
longest_episode = 0

for _ in tqdm(range(max_num_episodes)):
    episode_cost = 0
    episode_steps = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state, reparameterize=False)
        next_state, cost, terminated, truncated, _ = environment.step(action)

        done = terminated or truncated

        agent.store_transition(state, action, cost, next_state, done)

        state = next_state

        if done:
            break

        episode_cost += cost
        episode_steps += 1
        total_steps += 1

    state, _ = environment.reset()
    
    for j in range(episode_steps):
        agent.train()
    
    if episode_steps > longest_episode:
        longest_episode = episode_steps

    steps_per_episode.append(episode_steps)
    cost_arr.append(episode_cost)
    step_arr.append(total_steps)
    lambda_arr.append(agent.lamda.item())
    beta_arr.append(agent.beta.item())

np.save("data/cartpole/arrays/lac-cost2-arr.npy", np.array(cost_arr))
np.save("data/cartpole/arrays/lac-step2-arr.npy", np.array(step_arr))
np.save("data/cartpole/arrays/lac-lambda2-arr.npy", np.array(lambda_arr))
np.save("data/cartpole/arrays/lac-beta2-arr.npy", np.array(beta_arr))
print(f"Longest Episode: {longest_episode}")
print(f"Average Steps per Episode: {np.mean(steps_per_episode)}")
print(f"Beta: {agent.beta.item()}")
print(f"Lambda: {agent.lamda.item()}")
agent.save()