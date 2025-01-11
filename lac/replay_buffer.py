from collections import deque
import random
import numpy as np

class ReplayBuffer():
    def __init__(self, mem_length, finite_horizon, horizon_n):
        self.mem_length = int(mem_length)
        self.finite_horizon = finite_horizon
        self.horizon_n = horizon_n

        self.memory = deque(maxlen=self.mem_length)
        self.current_path = deque(maxlen=self.mem_length)
    
    def length(self):
        return len(self.memory)
    
    def store(self, data):
        self.current_path.append(data)
        state, action, reward, next_state, done = data

        if not done:
            return
        
        elif done and self.finite_horizon:
            rewards = [data[2] for data in self.current_path]
            rewards = np.array(rewards)
            last_reward = rewards[-1]

            rewards_aug = np.concatenate([rewards, last_reward*np.ones(self.horizon_n)])
            horizon_values = [np.sum(rewards_aug[i:i+self.horizon_n]) for i in range(len(rewards))]

            for tup, value in zip(self.current_path, horizon_values):
                self.memory.append((*tup, value))
            
            self.current_path.clear()


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    

