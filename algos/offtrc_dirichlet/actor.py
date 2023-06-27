from algos.common.network_base import MLP, initWeights
from algos.common.actor_base import ActorBase

import numpy as np
import torch

class Actor(ActorBase):
    def __init__(self, device:torch.device, n_assets:int, feature_dim:int, actor_cfg:dict) -> None:
        ActorBase.__init__(self, device)
        """
        state: # of assets x # of features (11 x 9 = 99)
        action: next portfolio weights (11)
        """
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.state_dim = n_assets * feature_dim
        self.action_dim = n_assets

        # for model
        """
        input: features -> concetration(=exp(scores)) -> concat(concentration) -> Dirichlet
        """
        activation_name = actor_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        self.add_module('model', MLP(
            input_size=self.feature_dim, output_size=1, \
            shape=actor_cfg['mlp']['shape'], activation=self.activation,
        ))
        if 'clip_range' in actor_cfg.keys():
            for item_idx in range(len(actor_cfg['clip_range'])):
                item = actor_cfg['clip_range'][item_idx]
                if type(item) == str:
                    actor_cfg['clip_range'][item_idx] = eval(item)
            self.clip_range = actor_cfg['clip_range']
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
        alphas = self.model(state.view(state.shape[:-1] + (self.n_assets, self.feature_dim))).squeeze(-1)
        alphas = torch.clamp(alphas, self.clip_range[0], self.clip_range[1])
        alphas = torch.exp(alphas) + 1.0
        return alphas

    def updateActionDist(self, state:torch.Tensor) -> None:
        """
        state: (batch_size, n_assets, feature_dim)
        alphas: (batch_size, n_assets)
        action_dist: (batch_size,)
        """
        self.alphas = self.forward(state)
        self.action_dist = torch.distributions.Dirichlet(self.alphas)

    def sample(self) -> torch.Tensor:
        self.sampled_action = self.action_dist.rsample()
        return self.sampled_action

    def getDist(self) -> torch.distributions.Distribution:
        return self.action_dist

    def getEntropy(self) -> torch.Tensor:
        return torch.mean(self.action_dist.entropy())

    def getLogProb(self) -> torch.Tensor:
        return self.action_dist.log_prob(self.sampled_action)

    def getConcentration(self) -> torch.Tensor:
        return self.alphas

    def initialize(self) -> None:
        self.apply(initWeights)
