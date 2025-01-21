# Run Cart Pole env
import torch

import gymnasium as gym
import env
from sac.agent import SACAgent
from lac.agent import LAC
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('CustomInvertedPendulum-v0', render_mode='human')
print(environment.action_space.high[0])
model = 'lac'

if model == 'lac':
    agent = LAC(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0], finite_horizon=True, horizon_n=5)
else: 
    agent = SACAgent(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0])
agent.load()

state, info = environment.reset(seed=42)
max_num_episodes = 10
max_episode_length = 250
cost_arr = []
step_arr = []
action_arr = []
value_arr = []
lyapunov_arr = []
steps_per_episode = []
total_steps = 0
longest_episode = 0

for _ in tqdm(range(max_num_episodes)):
    episode_cost = 0
    episode_steps = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state, reparameterize=False)
        action_arr.append(float(action))
        next_state, cost, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated

        state = next_state

        episode_cost += cost
        episode_steps += 1
        step_arr.append(episode_steps)
        cost_arr.append(episode_cost)

        if terminated:
            cost_aug = np.concatenate([cost_arr, cost*np.ones(max_episode_length - len(cost_arr))])
            break

    state,_ = environment.reset()
    print(f"Episode Length: {episode_steps}")
    print(f"Terminated: {terminated} ")
    print(f"Truncated: {truncated}")


plt.plot(step_arr, action_arr, marker='o', linestyle='-')   
plt.xlabel('Step')
plt.ylabel('Action')
plt.title('Control Policy over One Episode')
plt.show()