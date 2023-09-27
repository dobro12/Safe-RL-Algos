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
    def __init__(self, obs_dim:int, action_dim:int, len_replay_buffer:int, \
                discount_factor:float, gae_coeff:float, n_envs:int, \
                device:torch.device, n_steps:int, n_update_steps:int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.gae_coeff = gae_coeff
        self.n_envs = n_envs
        self.device = device
        self.n_steps = n_steps
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        self.n_update_steps = n_update_steps
        self.n_update_steps_per_env = int(self.n_update_steps/self.n_envs)
        self.len_replay_buffer = len_replay_buffer
        self.len_replay_buffer_per_env = int(self.len_replay_buffer/self.n_envs)

        self.storage = [deque(maxlen=self.len_replay_buffer_per_env) for _ in range(self.n_envs)]

    ################
    # Public Methods
    ################

    def addTransition(self, states, actions, means, stds, log_probs, rewards, costs, dones, fails, next_states):
        for env_idx in range(self.n_envs):
            self.storage[env_idx].append([
                states[env_idx], actions[env_idx], means[env_idx], stds[env_idx], log_probs[env_idx], 
                rewards[env_idx], costs[env_idx], dones[env_idx], fails[env_idx], next_states[env_idx]
            ])
    
    def getBatches(self, actor, reward_critic, cost_critic, cost_std_critic):
        state_len = len(self.storage[0])
        n_latest_steps = min(state_len, self.n_steps_per_env)
        n_update_steps = min(state_len, self.n_update_steps_per_env)

        # process the latest trajectories
        start_idx = state_len - n_latest_steps
        end_idx = start_idx + n_latest_steps
        states_list, actions_list, mu_means_list, mu_stds_list, reward_targets_list, cost_targets_list, cost_std_targets_list, \
            reward_gaes_list, cost_gaes_list, cost_var_gaes_list = \
                self._processBatches(actor, reward_critic, cost_critic, cost_std_critic, start_idx, end_idx)

        # process the rest trajectories
        if n_update_steps > n_latest_steps:
            start_idx = np.random.randint(0, state_len - n_update_steps + 1)
            end_idx = start_idx + n_update_steps - n_latest_steps
            temp_states_list, temp_actions_list, temp_mu_means_list, temp_mu_stds_list, temp_reward_targets_list, temp_cost_targets_list, temp_cost_std_targets_list, \
                temp_reward_gaes_list, temp_cost_gaes_list, temp_cost_var_gaes_list = \
                    self._processBatches(actor, reward_critic, cost_critic, cost_std_critic, start_idx, end_idx)
            
            states_list.extend(temp_states_list)
            actions_list.extend(temp_actions_list)
            mu_means_list.extend(temp_mu_means_list)
            mu_stds_list.extend(temp_mu_stds_list)
            reward_targets_list.extend(temp_reward_targets_list)
            cost_targets_list.extend(temp_cost_targets_list)
            cost_std_targets_list.extend(temp_cost_std_targets_list)
            reward_gaes_list.extend(temp_reward_gaes_list)
            cost_gaes_list.extend(temp_cost_gaes_list)
            cost_var_gaes_list.extend(temp_cost_var_gaes_list)

        # convert to tensor
        states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32)
        mu_means_tensor = torch.tensor(np.concatenate(mu_means_list, axis=0), device=self.device, dtype=torch.float32)
        mu_stds_tensor = torch.tensor(np.concatenate(mu_stds_list, axis=0), device=self.device, dtype=torch.float32)
        reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
        cost_targets_tensor = torch.tensor(np.concatenate(cost_targets_list, axis=0), device=self.device, dtype=torch.float32)
        cost_std_targets_tensor = torch.tensor(np.concatenate(cost_std_targets_list, axis=0), device=self.device, dtype=torch.float32)
        reward_gaes_tensor = torch.tensor(np.concatenate(reward_gaes_list, axis=0), device=self.device, dtype=torch.float32)
        cost_gaes_tensor = torch.tensor(np.concatenate(cost_gaes_list, axis=0), device=self.device, dtype=torch.float32)
        cost_var_gaes_tensor = torch.tensor(np.concatenate(cost_var_gaes_list, axis=0), device=self.device, dtype=torch.float32)

        cost_mean = cost_critic(states_tensor).mean().item()
        cost_var_mean = torch.square(cost_std_critic(states_tensor)).mean().item()

        return states_tensor, actions_tensor, mu_means_tensor, mu_stds_tensor, reward_targets_tensor, cost_targets_tensor, cost_std_targets_tensor, \
            reward_gaes_tensor, cost_gaes_tensor, cost_var_gaes_tensor, cost_mean, cost_var_mean
    
    #################
    # Private Methods
    #################

    def _processBatches(self, actor, reward_critic, cost_critic, cost_std_critic, start_idx, end_idx):
        states_list = []
        mu_means_list = []
        mu_stds_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_list = []
        cost_std_targets_list = []
        reward_gaes_list = []
        cost_gaes_list = []
        cost_var_gaes_list = []
        cost_mean_list = []
        cost_var_mean_list = []

        for env_idx in range(self.n_envs):
            env_trajs = list(self.storage[env_idx])[start_idx:end_idx]
            states = np.array([traj[0] for traj in env_trajs])
            actions = np.array([traj[1] for traj in env_trajs])
            mu_means = np.array([traj[2] for traj in env_trajs])
            mu_stds = np.array([traj[3] for traj in env_trajs])
            log_probs = np.array([traj[4] for traj in env_trajs])
            rewards = np.array([traj[5] for traj in env_trajs])
            costs = np.array([traj[6] for traj in env_trajs])
            dones = np.array([traj[7] for traj in env_trajs])
            fails = np.array([traj[8] for traj in env_trajs])
            next_states = np.array([traj[9] for traj in env_trajs])

            states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
            mu_log_probs_tensor = torch.tensor(log_probs, device=self.device, dtype=torch.float32)

            # for rho
            epsilons_tensor = torch.normal(mean=torch.zeros_like(actions_tensor), std=torch.ones_like(actions_tensor))
            actor.updateActionDist(states_tensor, epsilons_tensor)
            dists_tensor = actor.getDist()
            old_log_probs_tensor = torch.sum(dists_tensor.log_prob(actions_tensor), dim=-1)
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
                reward_targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_reward_values[t] \
                                + self.discount_factor*(1.0 - dones[t])*reward_delta
                cost_targets[t] = costs[t] + self.discount_factor*(1.0 - fails[t])*next_cost_values[t] \
                                + self.discount_factor*(1.0 - dones[t])*cost_delta
                cost_var_targets[t] = np.square(costs[t] + (1.0 - fails[t])*self.discount_factor*next_cost_values[t]) - np.square(cost_values[t]) + \
                                (1.0 - fails[t])*(self.discount_factor**2)*next_cost_var_values[t] + \
                                (1.0 - dones[t])*(self.discount_factor**2)*cost_var_delta
                reward_delta = self.gae_coeff*rhos[t]*(reward_targets[t] - reward_values[t])
                cost_delta = self.gae_coeff*rhos[t]*(cost_targets[t] - cost_values[t])
                cost_var_delta = self.gae_coeff*rhos[t]*(cost_var_targets[t] - cost_var_values[t])
            reward_gaes = reward_targets - reward_values
            cost_gaes = cost_targets - cost_values
            cost_var_gaes = cost_var_targets - cost_var_values
            cost_var_targets = np.clip(cost_var_targets, 0.0, np.inf)

            # append
            states_list.append(states)
            mu_means_list.append(mu_means)
            mu_stds_list.append(mu_stds)
            actions_list.append(actions)
            reward_targets_list.append(reward_targets)
            cost_targets_list.append(cost_targets)
            cost_std_targets_list.append(np.sqrt(cost_var_targets))
            reward_gaes_list.append(reward_gaes)
            cost_gaes_list.append(cost_gaes)
            cost_var_gaes_list.append(cost_var_gaes)
            cost_mean_list.append(np.mean(costs)/(1.0 - self.discount_factor))
            cost_var_mean_list.append(np.mean(cost_var_targets))

        return states_list, actions_list, mu_means_list, mu_stds_list, reward_targets_list, cost_targets_list, cost_std_targets_list, \
                reward_gaes_list, cost_gaes_list, cost_var_gaes_list
    
