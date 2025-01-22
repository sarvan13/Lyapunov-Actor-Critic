# Run Cart Pole env
import torch

import gymnasium as gym
import env
from lac.agent import LAC
from sac.agent import SACAgent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('CustomInvertedPendulum-v0', render_mode='human')
print(environment.action_space.high[0])
model = 'sac'

if model == 'lac':
    agent = LAC(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0])
    agent.load()
else:
    agent = SACAgent(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0])
    agent.load()

state, info = environment.reset(seed=42)
max_num_episodes = 10
max_episode_length = 250
cost_arr = []
step_arr = []
action_arr = []
steps_per_episode = []
total_steps = 0
longest_episode = 0

for _ in tqdm(range(max_num_episodes)):
    episode_cost = 0
    episode_steps = 0
    lyapunov_arr = []
    x = []
    theta = []
    for i in range(max_episode_length):
        action = agent.choose_action(state, reparameterize=False)
        # print(f"Lyapunov: {l_c.item()}")
        # print(f"X: {state[0]}")
        # print(f"Theta: {state[1]}")
        # print(f"Action: {action}")
        # input("Press Enter to take the next step...")
        if model == 'lac':
            l_net_out = agent.l_net.forward(torch.tensor([state], dtype=torch.float32).to(agent.policy.device), torch.tensor([action], dtype=torch.float32).to(agent.policy.device))
            l_c = (l_net_out ** 2).sum(dim=1)
            lyapunov_arr.append(l_c.item())
            action_arr.append(float(action))
            x.append(state[0])
            theta.append(state[1])

        # action_arr.append(float(action))
        next_state, cost, terminated, truncated, _ = environment.step(action)

        state = next_state

        # episode_cost += cost
        # episode_steps += 1
        # step_arr.append(episode_steps)
        # cost_arr.append(episode_cost)

        if terminated:
            break

    state, _ = environment.reset()

    if model == 'lac':
        plt.scatter(x, lyapunov_arr, marker='o')
        plt.xlabel('X')
        plt.ylabel('Lyapunov')
        plt.title('Lyapunov over One Episode')
        plt.show()

        plt.scatter(theta, lyapunov_arr, marker='o')
        plt.xlabel('Theta')
        plt.ylabel('Lyapunov')
        plt.title('Lyapunov over One Episode')
        plt.show()

# plt.plot(step_arr, action_arr, marker='o', linestyle='-')   
# plt.xlabel('Step')
# plt.ylabel('Action')
# plt.title('Control Policy over One Episode')
# plt.show()