import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from .replay_buffer import ReplayBuffer
from .networks import ActorNet, LyapunovCriticNet

class LAC():
    def __init__(self, state_dims, action_dims, max_action, alr=1e-4, clr=3e-4, llr=3e-4, gamma=0.99, tau=0.005, entropy=-1, 
                 alpha=1, mem_length=1e6, batch_size=256, finite_horizon=True, horizon_n=5):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.max_action = max_action
        self.alr = alr
        self.clr = clr
        self.llr = llr
        self.gamma = gamma
        self.tau = tau
        self.entropy = entropy
        self.alpha = alpha
        self.mem_length = mem_length
        self.batch_size = batch_size
        self.finite_horizon = finite_horizon
        self.horizon_n = horizon_n

        self.lamda = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.lagrange_optimizer = optim.Adam([self.beta, self.lamda], lr=self.clr)

        self.memory = ReplayBuffer(self.mem_length, self.finite_horizon, self.horizon_n)

        self.policy = ActorNet(state_dims, action_dims, max_action, lr=alr)
        self.l_net = LyapunovCriticNet(state_dims, action_dims, lr=llr)

        if not finite_horizon:
            self.l_target_net = LyapunovCriticNet(state_dims, action_dims, lr=llr)
            self.l_target_net.load_state_dict(self.l_net.state_dict())

    # Do not use for policy gradient calculations as it returns a numpy array array detached from the
    # backwards propogation graph
    def choose_action(self, state, reparameterize=False):
        state = torch.tensor([state], dtype=torch.float).to(self.policy.device)

        action, _ = self.policy.sample(state, reparameterize)

        return action.cpu().detach().numpy()[0]
    
    def train(self):
        if self.memory.length() < self.batch_size:
            return
        else:
            batch = self.memory.sample(self.batch_size)

            states, actions, rewards, next_states, dones, horizon_value = zip(*batch)

            states = torch.tensor(states, dtype=torch.float).to(self.policy.device)
            actions = torch.tensor(actions, dtype=torch.float).to(self.policy.device)
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.policy.device)
            next_states = torch.tensor(next_states, dtype=torch.float).to(self.policy.device)
            dones = torch.tensor(dones, dtype=torch.float).to(self.policy.device)
            horizon_values = torch.tensor(horizon_value, dtype=torch.float).to(self.policy.device)

            # Calculate L_c Lyapunov Loss
            l_net_out = self.l_net.forward(states, actions)
            l_c = (l_net_out ** 2).sum(dim=1)
            
            if self.finite_horizon:
                l_target = horizon_values
            else:
                next_actions = self.policy.sample(next_states, reparameterize=False)
                l_target = rewards + self.gamma * self.l_target_net.forward(next_states, next_actions)
            
            loss_func = nn.MSELoss()
            lyapunov_loss = loss_func(l_c,l_target)

            self.l_net.optimizer.zero_grad()
            lyapunov_loss.backward()
            self.l_net.optimizer.step()

            # Calculate Policy Loss
            _, log_probs = self.policy.sample(states, reparameterize=True)
            next_actions, _ = self.policy.sample(next_states, reparameterize=True)
            l_net_out_next = self.l_net.forward(next_states, next_actions)
            l_c_next = (l_net_out_next ** 2).sum(dim=1)
            l_net_out = self.l_net.forward(states, actions)
            l_c = (l_net_out ** 2).sum(dim=1)

            policy_loss = self.beta * (log_probs + self.entropy) + self.lamda * (l_c_next - l_c.detach() + self.alpha*rewards)
            policy_loss = policy_loss.mean()

            self.policy.optimizer.zero_grad()
            self.lagrange_optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()
            self.lagrange_optimizer.step()

            self.beta.data.clamp_(min=0, max=1)
            self.lamda.data.clamp_(min=0, max=1)


    def store_transition(self, state, action, reward, next_state, terminated):
        self.memory.store((state, action, reward, next_state, terminated))

    def save(self):
        self.policy.save()
        self.l_net.save()
    
    def load(self):
        self.policy.load()
        self.l_net.load()
        if not self.finite_horizon:
            self.l_target_net.load_state_dict(self.l_net.state_dict())