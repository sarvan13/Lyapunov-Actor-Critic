import torch
import torch.nn as nn
import numpy as np
import random
from networks import ActorNet, LyapunovCriticNet

class LAC():
    def __init__(self, state_dims, action_dims, max_action, alr=1e-4, clr=3e-4, llr=3e-4, gamma=0.99, tau=0.005, entropy=-1, 
                 alpha=1, mem_length=1e6, finite_horizon=True, horizon_n=5):
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
        self.finite_horizon = finite_horizon
        self.horizon_n = horizon_n

        self.policy = ActorNet(alr, state_dims, action_dims, max_action)
        self.l_net = LyapunovCriticNet(llr, state_dims)

        if not finite_horizon:
            self.l_target_net = LyapunovCriticNet(llr, state_dims)
            self.l_target_net.load_state_dict(self.l_net.state_dict())

        