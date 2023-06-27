from collections import deque
import ruamel.yaml as yaml
import numpy as np
import pmenv
import torch
import gym
import sys
import os

"""
Data Shape: (N, K, F)
N: # of data (days)
K: # of portfolios (1 cash + 10 stocks = 11)
F: # of indicators (8) + closed prices (1)
"""
dir_path = os.path.dirname(os.path.realpath(__file__))
train_data = np.load(f'{dir_path}/pmenv/data/train_data_tensor_10.npy')
test_data = np.load(f'{dir_path}/pmenv/data/test_data_tensor_10.npy')

class Env(pmenv.Environment):
    def __init__(self, data:np.ndarray=train_data, init_balance:float=1e6, holding_period:int=5):
        super().__init__(data)
        """
        self.K: # of portfolios (1 (cash) + # of stocks)
        self.F: # of indicators (except close price)
        self.balance: balance (cash)
        self.portfolio: current ratio of portfolios (order: cash, stock1, stock2, ...)
        """
        self.init_balance = init_balance
        self.holding_period = holding_period
        self.max_episode_len = int(data.shape[0] / holding_period)

    def getState(self, observation):
        """
        # of portfolios (11) x features (9 = indicators (8) + portfolio ratio (1))
        """
        portfolio = self.portfolio[:,np.newaxis]
        state = np.concatenate([observation, portfolio], axis=1)
        return state.ravel()

    def step(self, action):
        """
        Action: sampled portfolio - now portfolio
        """
        reward = 0.0
        for repeat_idx in range(self.holding_period):
            if repeat_idx == 0: temp_action = (action.ravel() - self.portfolio)[1:]
            else: temp_action = np.zeros(action.shape[0] - 1)
            temp_state, temp_reward, done = super().step(temp_action)
            assert temp_reward.shape == (1,)
            reward += temp_reward[0]
            if done: break
        state = self.getState(temp_state)
        return state, reward, done

    def reset(self):
        obs = super().reset(self.init_balance)
        obs = self.getState(obs)
        return obs
