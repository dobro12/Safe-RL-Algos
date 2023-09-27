from utils.vectorize import RunningMeanStd

from abc import ABC, abstractmethod
import numpy as np
import torch

class AgentBase(ABC):
    def __init__(
            self, name:str, 
            device:torch.device,
            obs_dim:int,
            action_dim:int,
            norm_obs:bool=False, 
            norm_reward:bool=False) -> None:
        # set attributes
        self.name = name
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # for normalization
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.obs_rms = RunningMeanStd('obs', self.obs_dim)
        self.reward_rms = RunningMeanStd('reward', 1)

    @abstractmethod
    def getAction(self, state:np.ndarray, deterministic:bool=False) -> np.ndarray:
        """
        Return action given state.
        if state's dimension is (batch_size, state_dim), then action's dimension is (batch_size, action_dim).
        if state's dimension is (state_dim), then action's dimension is (action_dim).
        if deterministic is True, then return deterministic action.
        """

    @abstractmethod
    def train(self) -> dict:
        """
        Update agent's parameters such as actor and critics.
        Return update information in dictionary format.
        ex) {"actor_loss":actor_loss, "critic_loss":critic_loss}
        """

    @abstractmethod
    def save(self) -> None:
        """
        Save agent's parameters such as actors, critics, and optimizers.
        """

    @abstractmethod
    def load(self) -> None:
        """
        Load agent's parameters such as actors, critics, and optimizers.
        """
