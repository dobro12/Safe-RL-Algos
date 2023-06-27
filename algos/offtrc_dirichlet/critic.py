from algos.common.network_base import MLP, initWeights
from algos.common.critic_base import CriticBase

import numpy as np
import torch

class Critic(CriticBase):
    def __init__(self, device:torch.device, n_assets:int, feature_dim:int, critic_cfg:dict) -> None:
        super().__init__(device)

        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.state_dim = n_assets * feature_dim

        activation_name = critic_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        self.add_module('model', MLP(
            input_size=self.feature_dim, output_size=1, \
            shape=critic_cfg['mlp']['shape'], activation=self.activation,
        ))
        self.add_module("decoder", torch.nn.Sequential(
            self.activation(),
            torch.nn.Linear(self.n_assets, 1),
        ))
        if 'out_activation' in critic_cfg.keys():
            self.out_activation = eval(f'torch.nn.functional.{critic_cfg["out_activation"]}')
        else:
            self.out_activation = lambda x: x
        if 'clip_range' in critic_cfg.keys():
            for item_idx in range(len(critic_cfg['clip_range'])):
                item = critic_cfg['clip_range'][item_idx]
                if type(item) == str:
                    critic_cfg['clip_range'][item_idx] = eval(item)
            self.clip_range = critic_cfg['clip_range']
        else:
            self.clip_range = [-np.inf, np.inf]

    ################
    # public methods
    ################

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        """
        state: (batch_size, n_assets * feature_dim)
        output: concentration (batch_size, n_assets)
        """
        x = self.model(state.view(state.shape[:-1] + (self.n_assets, self.feature_dim))).squeeze(-1)
        x = self.decoder(x).squeeze(-1)
        x = self.out_activation(x)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x

    def getLoss(self, state:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """
        state: (batch_size, n_assets * feature_dim)
        target: (batch_size,)
        """
        return torch.nn.functional.mse_loss(self.forward(state), target)

    def initialize(self) -> None:
        self.apply(initWeights)
