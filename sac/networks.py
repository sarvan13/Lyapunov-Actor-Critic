import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal

class ActorNet(nn.Module):
    def __init__(self, lr, state_dims, action_dims, max_action, fc1_dims=256, fc2_dims=256, 
                 reparam_noise=1e-6, name='Actor', save_dir='tmp'):
        super(ActorNet, self).__init__()
        self.lr = lr
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.reparam_noise = reparam_noise
        self.name = name
        self.save_path = os.path.join(save_dir, name)

        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.action_dims)
        self.log_sigma = nn.Linear(self.fc2_dims, self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        layer1 = torch.relu(self.fc1(state))
        layer2 = torch.relu(self.fc2(layer1))
        mean = self.mu(layer2)
        log_std = self.log_sigma(layer2).clamp(-20,2)
        std = torch.exp(log_std)

        return mean, std
    
    def sample(self, state, reparameterize=True):
        mean, std = self.forward(state)
        normal = Normal(mean, std)

        if reparameterize:
            sampled_action = normal.rsample()
        else:
            sampled_action = normal.sample()

        tanh_action = torch.tanh(sampled_action)
        action = tanh_action * self.max_action
        log_prob = normal.log_prob(sampled_action) - torch.log(self.max_action * (1 - tanh_action.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob
    
    def save(self):
        torch.save(self.state_dict(), self.save_path)
    
    def load(self):
        self.load_state_dict(torch.load(self.save_path))
    
class QNet(nn.Module):
    def __init__(self, lr, state_dims, action_dims, fc1_dims=256, fc2_dims=256, 
                 name='Q-Critic', save_dir='tmp'):
        super(QNet, self).__init__()
        self.lr = lr
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.save_path = os.path.join(save_dir, name)

        self.fc1 = nn.Linear(self.state_dims + self.action_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        layer1 = torch.relu(self.fc1(state_action))
        layer2 = torch.relu(self.fc2(layer1))
        q_value = self.q(layer2)

        return q_value
    
    def save(self):
        torch.save(self.state_dict(), self.save_path)
    
    def load(self):
        self.load_state_dict(torch.load(self.save_path))

class ValueNet(nn.Module):
    def __init__(self, lr, state_dims, fc1_dims=256, fc2_dims=256, name='Value-Net', save_dir='tmp'):
        super(ValueNet, self).__init__()
        self.lr = lr
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.save_path = os.path.join(self.name, save_dir)

        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.value = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = torch.relu(self.fc1(state))
        layer2 = torch.relu(self.fc2(layer1))
        value = self.value(layer2)

        return value
    
    def save(self):
        torch.save(self.state_dict(), self.save_path)
    
    def load(self):
        self.load_state_dict(torch.load(self.save_path))