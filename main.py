# Run Cart Pole env
import torch
torch.autograd.set_detect_anomaly(True)

import gymnasium as gym
import env
from lac.agent import LAC
from tqdm import tqdm
import numpy as np

environment = gym.make('CustomInvertedPendulum-v0')

agent = LAC(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high[0], finite_horizon=True, horizon_n=5)

state, info = environment.reset(seed=42)
max_num_episodes = 1000
max_episode_length = 250
cost_arr = []
step_arr = []
total_steps = 0

for _ in tqdm(range(max_num_episodes)):
    episode_cost = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state)
        next_state, cost, terminated, truncated, _  = environment.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, cost, next_state, done)

        episode_cost += cost
        total_steps += 1

        state = next_state

        #environment.render()

        if terminated:
            break

    state, _ = environment.reset()
     
    for j in range(i):
        agent.train()
    
    cost_arr.append(episode_cost)
    step_arr.append(total_steps)

print(f"Beta: {agent.beta}")
print(f"Lambda: {agent.lamda}")
np.save("lac-cost-arr.npy", np.array(cost_arr))
np.save("lac-step-arr.npy", np.array(step_arr))
agent.save()