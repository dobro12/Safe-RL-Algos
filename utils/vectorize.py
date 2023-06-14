from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from collections import deque
import numpy as np
import pickle
import wandb
import sys
import os


class CustomSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, start_method=None):
        super().__init__(env_fns, start_method)
        self.obs_rms = RunningMeanStd(self.observation_space.shape[0])

    def reset(self):
        observations = super().reset()
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        return norm_observations

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        for info in infos:
            if 'terminal_observation' in info.keys():
                info['terminal_observation'] = self.obs_rms.normalize(info['terminal_observation'])
        return norm_observations, rewards, dones, infos
    
    def loadScaling(self, save_dir, model_num):
        self.obs_rms.load(save_dir, model_num)
    
    def saveScaling(self, save_dir, model_num):
        self.obs_rms.save(save_dir, model_num)


class RunningMeanStd(object):
    def __init__(self, state_dim, limit_cnt=np.inf):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: (float) helps with arithmetic issues
        :param state_dim: (int) the state_dim of the data stream's output
        """
        self.limit_cnt = limit_cnt
        self.mean = np.zeros(state_dim, np.float32)
        self.var = np.ones(state_dim, np.float32)
        self.count = 0.0

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        return

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if self.count >= self.limit_cnt: return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        return

    def normalize(self, observations):
        return (observations - self.mean)/np.sqrt(self.var + 1e-8)
    
    def load(self, save_dir, model_num):
        file_name = f"{save_dir}/checkpoint/scale_{model_num}.pkl"
        with open(file_name, 'rb') as f:
            self.mean, self.var, self.count = pickle.load(f)

    def save(self, save_dir, model_num):
        file_name = f"{save_dir}/checkpoint/scale_{model_num}.pkl"
        with open(file_name, 'wb') as f:
            pickle.dump([self.mean, self.var, self.count], f)

