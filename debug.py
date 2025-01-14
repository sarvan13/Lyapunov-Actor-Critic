# Run Cart Pole env
import torch
torch.autograd.set_detect_anomaly(True)

import gymnasium as gym
import env
from lac.agent import LAC
from tqdm import tqdm
import numpy as np

environment = gym.make("CartPoleadv-v1")

agent = LAC(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0], finite_horizon=True, horizon_n=5)

state, info = environment.reset(seed=42)
max_num_episodes = 5000
max_episode_length = 100
num_gradient_updates = 1
cost_arr = []
step_arr = []
total_steps = 0
k = 0
for _ in tqdm(range(max_num_episodes)):
    episode_cost = 0
    episode_steps = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state)
        next_state, cost, terminated, truncated = environment.step(action)

        agent.store_transition(state, action, cost, next_state, terminated)

        episode_cost += cost
        episode_steps += 1

        #environment.render()

        if k > 50:
            l_out = agent.l_net.forward_single(torch.tensor(state, dtype=torch.float).to(agent.policy.device), torch.tensor(action, dtype=torch.float).to(agent.policy.device))
            l_c = (l_out ** 2).sum()
            print(f"Lyapunov Value: {l_c}")

        if terminated:
            break

    environment.reset()
    total_steps += episode_steps
    
    for j in range(num_gradient_updates):
        agent.train()
    
    # print("Episode Cost:" + str(episode_cost))
    # print("Episode Steps:" + str(episode_steps))
    k += 1
    print()