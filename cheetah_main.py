# Run Cart Pole env
import torch

import gymnasium as gym
import env
from lac.agent import LAC
from sac.agent import SACAgent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('HalfCheetahCost-v0', render_mode='human')
print(environment.action_space.high[0])
model = 'lac'

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

        next_state, cost, terminated, _ = environment.step(action)

        state = next_state

        if terminated:
            break

    state, _ = environment.reset()