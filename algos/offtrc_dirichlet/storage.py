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
    def __init__(self, device:torch.device, len_replay_buffer:int, \
                discount_factor:float, gae_coeff:float, \
                n_steps:int, n_update_steps:int) -> None:
        self.device = device
        self.len_replay_buffer = len_replay_buffer
        self.discount_factor = discount_factor
        self.gae_coeff = gae_coeff
        self.n_steps = n_steps
        self.n_update_steps = n_update_steps
        self.storage = deque(maxlen=self.len_replay_buffer)

    ################
    # Public Methods
    ################

    def addTransition(self, state, action, concentration, log_prob, reward, cost, done, next_state):
        self.storage.append([state, action, concentration, log_prob, reward, cost, done, next_state])
    
    def getBatches(self, actor, reward_critic, cost_critic, cost_std_critic):
        state_len = len(self.storage)
        n_latest_steps = min(state_len, self.n_steps)
        n_update_steps = min(state_len, self.n_update_steps)

        # process the latest trajectories
        start_idx = state_len - n_latest_steps
        end_idx = start_idx + n_latest_steps
        states, actions, concentrations, reward_targets, cost_targets, cost_std_targets, \
            reward_gaes, cost_gaes, cost_var_gaes = \
                self._processBatches(actor, reward_critic, cost_critic, cost_std_critic, start_idx, end_idx)

        # process the rest trajectories
        if n_update_steps > n_latest_steps:
            start_idx = np.random.randint(0, state_len - n_update_steps + 1)
            end_idx = start_idx + n_update_steps - n_latest_steps
            temp_states, temp_actions, temp_concentrations, temp_reward_targets, temp_cost_targets, temp_cost_std_targets, \
                temp_reward_gaes, temp_cost_gaes, temp_cost_var_gaes = \
                    self._processBatches(actor, reward_critic, cost_critic, cost_std_critic, start_idx, end_idx)
            
            states = np.concatenate([states, temp_states], axis=0)
            actions = np.concatenate([actions, temp_actions], axis=0)
            concentrations = np.concatenate([concentrations, temp_concentrations], axis=0)
            reward_targets = np.concatenate([reward_targets, temp_reward_targets], axis=0)
            cost_targets = np.concatenate([cost_targets, temp_cost_targets], axis=0)
            cost_std_targets = np.concatenate([cost_std_targets, temp_cost_std_targets], axis=0)
            reward_gaes = np.concatenate([reward_gaes, temp_reward_gaes], axis=0)
            cost_gaes = np.concatenate([cost_gaes, temp_cost_gaes], axis=0)
            cost_var_gaes = np.concatenate([cost_var_gaes, temp_cost_var_gaes], axis=0)

        # convert to tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        concentrations_tensor = torch.tensor(concentrations, device=self.device, dtype=torch.float32)
        reward_targets_tensor = torch.tensor(reward_targets, device=self.device, dtype=torch.float32)
        cost_targets_tensor = torch.tensor(cost_targets, device=self.device, dtype=torch.float32)
        cost_std_targets_tensor = torch.tensor(cost_std_targets, device=self.device, dtype=torch.float32)
        reward_gaes_tensor = torch.tensor(reward_gaes, device=self.device, dtype=torch.float32)
        cost_gaes_tensor = torch.tensor(cost_gaes, device=self.device, dtype=torch.float32)
        cost_var_gaes_tensor = torch.tensor(cost_var_gaes, device=self.device, dtype=torch.float32)

        cost_mean = cost_critic(states_tensor).mean().item()
        cost_var_mean = torch.square(cost_std_critic(states_tensor)).mean().item()

        return states_tensor, actions_tensor, concentrations_tensor, reward_targets_tensor, cost_targets_tensor, cost_std_targets_tensor, \
            reward_gaes_tensor, cost_gaes_tensor, cost_var_gaes_tensor, cost_mean, cost_var_mean
    
    #################
    # Private Methods
    #################

    def _processBatches(self, actor, reward_critic, cost_critic, cost_std_critic, start_idx, end_idx):
        env_trajs = list(self.storage)[start_idx:end_idx]
        states = np.array([traj[0] for traj in env_trajs])
        actions = np.array([traj[1] for traj in env_trajs])
        concentrations = np.array([traj[2] for traj in env_trajs])
        log_probs = np.array([traj[3] for traj in env_trajs])
        rewards = np.array([traj[4] for traj in env_trajs])
        costs = np.array([traj[5] for traj in env_trajs])
        dones = np.array([traj[6] for traj in env_trajs])
        next_states = np.array([traj[7] for traj in env_trajs])

        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        mu_log_probs_tensor = torch.tensor(log_probs, device=self.device, dtype=torch.float32)

        # for rho
        actor.updateActionDist(states_tensor)
        dists_tensor = actor.getDist()
        old_log_probs_tensor = dists_tensor.log_prob(actions_tensor)
        rhos_tensor = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 1.0)
        rhos = rhos_tensor.detach().cpu().numpy()

        # get values
        next_reward_values = reward_critic(next_states_tensor).detach().cpu().numpy()
        reward_values = reward_critic(states_tensor).detach().cpu().numpy()
        next_cost_values = cost_critic(next_states_tensor).detach().cpu().numpy()
        cost_values = cost_critic(states_tensor).detach().cpu().numpy()
        next_cost_var_values = torch.square(cost_std_critic(next_states_tensor)).detach().cpu().numpy()
        cost_var_values = torch.square(cost_std_critic(states_tensor)).detach().cpu().numpy()

        # get targets
        reward_delta = 0.0
        cost_delta = 0.0
        cost_var_delta = 0.0
        reward_targets = np.zeros_like(rewards) # n_steps
        cost_targets = np.zeros_like(costs) # n_steps
        cost_var_targets = np.zeros_like(costs) # n_steps
        for t in reversed(range(len(reward_targets))):
            reward_targets[t] = rewards[t] + self.discount_factor*next_reward_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*reward_delta
            cost_targets[t] = costs[t] + self.discount_factor*next_cost_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*cost_delta
            cost_var_targets[t] = np.square(costs[t] + self.discount_factor*next_cost_values[t]) - np.square(cost_values[t]) \
                            + (self.discount_factor**2)*next_cost_var_values[t] + \
                            (1.0 - dones[t])*(self.discount_factor**2)*cost_var_delta
            reward_delta = self.gae_coeff*rhos[t]*(reward_targets[t] - reward_values[t])
            cost_delta = self.gae_coeff*rhos[t]*(cost_targets[t] - cost_values[t])
            cost_var_delta = self.gae_coeff*rhos[t]*(cost_var_targets[t] - cost_var_values[t])
        cost_std_targets = np.sqrt(np.clip(cost_var_targets, 0.0, np.inf))
        reward_gaes = reward_targets - reward_values
        cost_gaes = cost_targets - cost_values
        cost_var_gaes = cost_var_targets - cost_var_values

        return states, actions, concentrations, reward_targets, cost_targets, cost_std_targets, \
                reward_gaes, cost_gaes, cost_var_gaes
    
