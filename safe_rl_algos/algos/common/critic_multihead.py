from algos.common.critic_base import (
    CriticS, CriticSA
)
from algos.common.network_base import MLP

import numpy as np
import torch


class CriticSMultiHead(CriticS):
    def __init__(
            self, device:torch.device, 
            state_dim:int, 
            reward_dim:int, 
            critic_cfg:dict) -> None:

        self.reward_dim = reward_dim
        super().__init__(device, state_dim, critic_cfg)

    def build(self) -> None:
        self.add_module('model', MLP(
            input_size=self.state_dim, output_size=self.reward_dim, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        x = self.model(state)
        x = self.last_activation(x)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x


class CriticSAMultiHead(CriticSA):
    def __init__(
            self, device:torch.device, 
            state_dim:int, 
            action_dim:int, 
            reward_dim:int, 
            critic_cfg:dict) -> None:

        self.reward_dim = reward_dim
        super().__init__(device, state_dim, action_dim, critic_cfg)

    def build(self) -> None:
        self.add_module('model', MLP(
            input_size=self.state_dim + self.action_dim, output_size=self.reward_dim, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.model(x)
        x = self.last_activation(x)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x
