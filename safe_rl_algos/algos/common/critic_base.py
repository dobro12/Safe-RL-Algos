from algos.common.network_base import MLP, initWeights

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import torch


class CriticBase(ABC, torch.nn.Module):
    def __init__(self, device:torch.device) -> None:
        torch.nn.Module.__init__(self)
        self.device = device

    @abstractmethod
    def getLoss(self) -> torch.Tensor:
        """
        Return action entropy given state.
        If state is None, use the internal state set in the `sample` function.
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize actor's parameters.
        """

class CriticS(CriticBase):
    def __init__(
            self, device:torch.device, 
            state_dim:int, 
            critic_cfg:dict) -> None:
        super().__init__(device)

        self.state_dim = state_dim
        self.critic_cfg = critic_cfg

        # for model
        self.activation = eval(f'torch.nn.{self.critic_cfg["mlp"]["activation"]}')
        if 'last_activation' in self.critic_cfg.keys():
            self.last_activation = eval(f'torch.nn.functional.{self.critic_cfg["last_activation"]}')
        else:
            self.last_activation = lambda x: x
        for item_idx in range(len(self.critic_cfg['clip_range'])):
            item = self.critic_cfg['clip_range'][item_idx]
            if type(item) == str:
                self.critic_cfg['clip_range'][item_idx] = eval(item)
        self.clip_range = self.critic_cfg['clip_range']

        # build model
        self.build()

    def build(self) -> None:
        self.add_module('model', MLP(
            input_size=self.state_dim, output_size=1, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        x = self.model(state)
        x = torch.squeeze(x, dim=-1)
        x = self.last_activation(x)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x

    def getLoss(self, state:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(self.forward(state), target)
        # return torch.nn.functional.mse_loss(self.forward(state), target)
    
    def initialize(self) -> None:
        self.apply(initWeights)


class CriticSA(CriticBase):
    def __init__(
            self, device:torch.device, 
            state_dim:int, 
            action_dim:int, 
            critic_cfg:dict) -> None:
        super().__init__(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_cfg = critic_cfg

        # for model
        self.activation = eval(f'torch.nn.{self.critic_cfg["mlp"]["activation"]}')
        if 'last_activation' in self.critic_cfg.keys():
            self.last_activation = eval(f'torch.nn.functional.{self.critic_cfg["last_activation"]}')
        else:
            self.last_activation = lambda x: x
        for item_idx in range(len(self.critic_cfg['clip_range'])):
            item = self.critic_cfg['clip_range'][item_idx]
            if type(item) == str:
                self.critic_cfg['clip_range'][item_idx] = eval(item)
        self.clip_range = self.critic_cfg['clip_range']

        # build model
        self.build()

    def build(self) -> None:
        self.add_module('model', MLP(
            input_size=self.state_dim + self.action_dim, output_size=1, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.model(x)
        x = torch.squeeze(x, dim=-1)
        x = self.last_activation(x)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x

    def getLoss(self, state:torch.Tensor, action:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(self.forward(state, action), target)
        # return torch.nn.functional.mse_loss(self.forward(state, action), target)
    
    def initialize(self) -> None:
        self.apply(initWeights)
