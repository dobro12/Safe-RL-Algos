from collections import deque
from copy import deepcopy
import numpy as np
import ctypes
import torch
import os

EPS = 1e-8 

def ctype_arr_convert(arr):
    arr = np.ravel(arr)
    return (ctypes.c_double * len(arr))(*arr)


class ReplayBuffer:
    def __init__(self, obs_dim:int, action_dim:int, \
                discount_factor:float, gae_coeff:float, n_envs:int, \
                device:torch.device, n_steps:int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.gae_coeff = gae_coeff
        self.n_envs = n_envs
        self.device = device
        self.n_steps = n_steps
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        self.storage = [[] for _ in range(self.n_envs)]

    ################
    # Public Methods
    ################

    def addTransition(self, states, actions, rewards, costs, dones, fails, next_states):
        for env_idx in range(self.n_envs):
            self.storage[env_idx].append([
                states[env_idx], actions[env_idx], rewards[env_idx], costs[env_idx], 
                dones[env_idx], fails[env_idx], next_states[env_idx]
            ])
    
    def getBatches(self, reward_critic, cost_critic):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_list = []
        reward_gaes_list = []
        cost_gaes_list = []
        constraint_list = []

        for env_idx in range(self.n_envs):
            env_trajs = self.storage[env_idx]
            states = np.array([traj[0] for traj in env_trajs])
            actions = np.array([traj[1] for traj in env_trajs])
            rewards = np.array([traj[2] for traj in env_trajs])
            costs = np.array([traj[3] for traj in env_trajs])
            dones = np.array([traj[4] for traj in env_trajs])
            fails = np.array([traj[5] for traj in env_trajs])
            next_states = np.array([traj[6] for traj in env_trajs])

            states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)

            # get values
            next_reward_values = reward_critic(next_states_tensor).detach().cpu().numpy()
            reward_values = reward_critic(states_tensor).detach().cpu().numpy()
            next_cost_values = cost_critic(next_states_tensor).detach().cpu().numpy()
            cost_values = cost_critic(states_tensor).detach().cpu().numpy()

            # get targets
            reward_delta = 0.0
            cost_delta = 0.0
            reward_targets = np.zeros_like(rewards) # n_steps
            cost_targets = np.zeros_like(costs) # n_steps
            for t in reversed(range(len(reward_targets))):
                reward_targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_reward_values[t] \
                                + self.discount_factor*(1.0 - dones[t])*reward_delta
                cost_targets[t] = costs[t] + self.discount_factor*(1.0 - fails[t])*next_cost_values[t] \
                                + self.discount_factor*(1.0 - dones[t])*cost_delta
                reward_delta = self.gae_coeff*(reward_targets[t] - reward_values[t])
                cost_delta = self.gae_coeff*(cost_targets[t] - cost_values[t])
            reward_gaes = reward_targets - reward_values
            cost_gaes = cost_targets - cost_values

            # append
            states_list.append(states)
            actions_list.append(actions)
            reward_targets_list.append(reward_targets)
            cost_targets_list.append(cost_targets)
            reward_gaes_list.append(reward_gaes)
            cost_gaes_list.append(cost_gaes)
            constraint_list.append(np.mean(costs)/(1.0 - self.discount_factor))

        # reset storage
        self.storage = [[] for _ in range(self.n_envs)]

        # convert to tensor
        states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32)
        reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
        cost_targets_tensor = torch.tensor(np.concatenate(cost_targets_list, axis=0), device=self.device, dtype=torch.float32)
        reward_gaes_tensor = torch.tensor(np.concatenate(reward_gaes_list, axis=0), device=self.device, dtype=torch.float32)
        cost_gaes_tensor = torch.tensor(np.concatenate(cost_gaes_list, axis=0), device=self.device, dtype=torch.float32)
        constraint = np.mean(constraint_list)

        return states_tensor, actions_tensor, reward_targets_tensor, cost_targets_tensor, \
            reward_gaes_tensor, cost_gaes_tensor, constraint
    
