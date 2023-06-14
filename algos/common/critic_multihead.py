from algos.common.critic_base import CriticSA
from algos.common.network_base import MLP

import numpy as np
import torch

class CriticSAMultiHead(CriticSA):
    def __init__(self, device:torch.device, state_dim:int, action_dim:int, n_rewards:int, critic_cfg:dict) -> None:
        self.n_rewards = n_rewards
        super().__init__(device, state_dim, action_dim, critic_cfg)

    def build(self) -> None:
        activation_name = self.critic_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        self.add_module('model', MLP(
            input_size=self.state_dim + self.action_dim, output_size=self.n_rewards, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))
        for item_idx in range(len(self.critic_cfg['clip_range'])):
            item = self.critic_cfg['clip_range'][item_idx]
            if type(item) == str:
                self.critic_cfg['clip_range'][item_idx] = eval(item)
        self.clip_range = self.critic_cfg['clip_range']

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.model(x)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x
