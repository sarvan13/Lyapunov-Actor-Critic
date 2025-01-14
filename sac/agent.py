import torch
import torch.optim as optim
import torch.nn as nn
from .networks import ActorNet, QNet, ValueNet
import gymnasium as gym
from collections import deque, namedtuple
import random

class SACAgent():
    def __init__(self, state_dims, action_dims, max_action, alr=1e-4, qlr=3e-4, vlr=3e-4, batch_size=256,
                 rewards_scale = 2, alpha = 0.2, gamma=1, tau=0.005, mem_length=1e5):
        self.actor = ActorNet(alr,state_dims, action_dims, max_action)
        self.q = QNet(qlr, state_dims, action_dims)
        self.value = ValueNet(vlr, state_dims)
        self.value_target = ValueNet(vlr, state_dims)
        self.value_target.load_state_dict(self.value.state_dict())

        self.max_action = max_action
        
        self.mem_length = mem_length
        self.replay_buffer = []
        # self.data_point = namedtuple()
        # self.replay_buffer = deque()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.rewards_scale = rewards_scale

        self.loss = nn.MSELoss()

    def save(self):
        self.actor.save()
        self.value.save()
        self.value_target.save()
        self.q.save()
    def load(self):
        self.actor.load()
        self.value.load()
        self.value_target.load()
        self.q.load()

    def remember(self, data_point):
        self.replay_buffer.append(data_point)
        if len(self.replay_buffer) > self.mem_length:
            self.replay_buffer.pop(0)

    def choose_action(self, state, reparameterize=False):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        action, _ = self.actor.sample(state, reparameterize)

        return action.cpu().detach().numpy()[0]
    
    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.actor.device)

        # Train Value Network
        # error = V(s) - E(Q(s,a) - log(pi(a|s)))
        self.value.optimizer.zero_grad()
        v = self.value.forward(states).view(-1)
        sampled_actions, log_probs = self.actor.sample(states, reparameterize=False)
        q_v = self.q.forward(states, sampled_actions).view(-1)
        target_v = q_v - log_probs.view(-1)
        v_loss = 0.5*self.loss(v, target_v.detach())
        v_loss.backward()
        self.value.optimizer.step()

        #Train Actor Network
        # error = log(pi(a|s)) - q(s,a)
        self.actor.optimizer.zero_grad()
        sampled_actions, log_probs = self.actor.sample(states, reparameterize=True)
        q_actor = self.q.forward(states, sampled_actions).view(-1)
        actor_loss = (log_probs.view(-1) - q_actor).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Train Q Network
        # error = Q(s,a) - (r(s,a) + gamma* E(V'(s)))
        self.q.optimizer.zero_grad()
        q = self.q.forward(states, actions).view(-1)
        next_value = self.value_target.forward(next_states).view(-1)
        next_value = (1 - dones).view(-1) * next_value
        q_target = self.rewards_scale*rewards.view(-1) + self.gamma * next_value
        q_loss = 0.5*self.loss(q,q_target.detach())

        q_loss.backward()
        self.q.optimizer.step()

        # Update V Target Network
        # Update the target value network

        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



