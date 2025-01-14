import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco import InvertedPendulumEnv

class CustomInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self, max_x=10.0, max_theta=0.2):
        super().__init__()
        self.max_x = max_x
        self.max_theta = max_theta

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        x = obs[0]  # Cart position
        theta = obs[1]  # Pendulum angle (radians)

        # Compute custom reward
        reward = (x / self.max_x) ** 2 + 20 * (theta / self.max_theta) ** 2

        if np.abs(x) >= self.max_x or np.abs(theta) >= self.max_theta:
            terminated = True
            reward = 100

        return obs, reward, terminated, truncated, info
