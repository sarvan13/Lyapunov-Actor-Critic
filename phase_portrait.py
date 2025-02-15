import gymnasium as gym
import numpy as np
from sac.agent import SACAgent
import matplotlib.pyplot as plt
import torch as T
import copy

env = gym.make('Pendulum-v1')
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = SACAgent(action_dims=env.action_space.shape[0],
                state_dims=env.observation_space.shape[0],
                max_action=env.action_space.high)
agent.load()
# n_games = 2
# theta_arr = []
# theta_dot_arr = []
# 
# for i in range(n_games):
#     print('Game:', i)
#     observation, _ = env.reset()
#     done = False
#     while not done:
#         action, prob, val = agent.choose_action(observation)
#         observation_, reward, terminated, truncated, info = env.step(action)
#         cos_theta, sin_theta, theta_dot = observation_
#         theta_arr.append(np.arctan2(sin_theta, cos_theta))
#         theta_dot_arr.append(theta_dot)
#         done = terminated or truncated
#         observation = observation_

# Parameters
num_episodes = 5  # Number of episodes to simulate
steps_per_episode = 200  # Number of steps per episode

# Function to run an episode and collect data
def run_episode():
    obs, _ = env.reset()
    theta_list = []
    theta_dot_list = []
    for _ in range(steps_per_episode):
        action = agent.choose_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)
        theta_list.append(theta)
        theta_dot_list.append(theta_dot)
        if terminated or truncated:
            break
    return theta_list, theta_dot_list

# Run multiple episodes and collect trajectories
trajectories = [run_episode() for _ in range(num_episodes)]

env.close()

# Plot trajectories
plt.figure(figsize=(10, 5))
for i, (theta_list, theta_dot_list) in enumerate(trajectories):
    plt.plot(theta_list, theta_dot_list, label=f'Episode {i+1}')
plt.xlabel("Theta (rad)")
plt.ylabel("Theta dot (rad/s)")
plt.title("Phase Portraits of Pendulum for SAC")
plt.legend()
plt.grid(True)
plt.show()