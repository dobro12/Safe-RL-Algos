from gymnasium.vector.utils.spaces import batch_space
from gymnasium.vector import AsyncVectorEnv
from gymnasium.utils import EzPickle

from typing import Iterator
import numpy as np
import pickle
import os


class ConAsyncVectorEnv(AsyncVectorEnv, EzPickle):
    """Vectorized environment that simultaneously runs multiple environments."""

    def __init__(
        self,
        env_fns: Iterator[callable],
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: env constructors
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.
        """
        super().__init__(env_fns, copy=copy)
        EzPickle.__init__(self, env_fns, copy=copy)

        # Get the reward space
        dummy_env = env_fns[0]()
        cost_space = dummy_env.unwrapped.cost_space
        self.cost_space = batch_space(cost_space, n=self.num_envs)
        self.single_cost_space = cost_space
        dummy_env.close()
        del dummy_env


class RunningMeanStd(object):
    def __init__(self, name:str, state_dim:int, limit_cnt:float=np.inf):
        self.name = name
        self.limit_cnt = limit_cnt
        self.mean = np.zeros(state_dim, np.float32)
        self.var = np.ones(state_dim, np.float32)
        self.count = 0.0

    def update(self, raw_data):
        arr = raw_data.reshape(-1, self.mean.shape[0])
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

    def normalize(self, observations, mean=0.0, std=1.0):
        norm_obs = (observations - self.mean)/np.sqrt(self.var + 1e-8)
        return norm_obs * std + mean
    
    def load(self, save_dir, model_num):
        file_name = f"{save_dir}/{self.name}_scale/{model_num}.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                self.mean, self.var, self.count = pickle.load(f)

    def save(self, save_dir, model_num):
        file_name = f"{save_dir}/{self.name}_scale/{model_num}.pkl"
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'wb') as f:
            pickle.dump([self.mean, self.var, self.count], f)
