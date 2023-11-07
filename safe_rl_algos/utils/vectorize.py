from gymnasium.vector.utils.spaces import batch_space
from gymnasium.vector.utils import concatenate
from gymnasium.vector import AsyncVectorEnv
from gymnasium.utils import EzPickle
import gymnasium

from typing import (
    Iterator, Any, List, Optional, Tuple, Union
)
from copy import deepcopy
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

    def reset(
        self,
        n_actions_per_env: Optional[List[int]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        observations = super().reset(seed=seed, options=options)[0]

        if n_actions_per_env is not None:
            observation_list = []

            for env_idx, pipe in enumerate(self.parent_pipes):
                n_actions = n_actions_per_env[env_idx]
                pipe = self.parent_pipes[env_idx]

                for _ in range(n_actions):
                    action = self.single_action_space.sample()
                    pipe.send(("step", action))
                    result, success = pipe.recv()
                    assert success
                    observation = result[0]

                observation_list.append(observation)

            if not self.shared_memory:
                self.observations = concatenate(
                    self.single_observation_space,
                    observation_list,
                    self.observations,
                )

            return deepcopy(self.observations) if self.copy else self.observations, {}
        else:
            return observations, {}


class NormObsWrapper(gymnasium.Wrapper):
    def __init__(self, env:gymnasium.Env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            obs_dim = env.single_observation_space.shape[0]
        else:
            obs_dim = env.observation_space.shape[0]
        self.obs_rms = RunningMeanStd("env_obs", obs_dim)

    def step(self, action, update_statistics:bool=True):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = self.normalize(obs, update_statistics)
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        if 'update_statistics' in kwargs.keys():
            update_statistics = kwargs['update_statistics']
            del kwargs['update_statistics']
        else:
            update_statistics = True

        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs, update_statistics), info

    def normalize(self, obs, update_statistics:bool=True):
        """Normalises the observation using the running mean and variance of the observations."""
        if update_statistics:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)
    
    def loadScaling(self, save_dir, model_num):
        self.obs_rms.load(save_dir, model_num)
    
    def saveScaling(self, save_dir, model_num):
        self.obs_rms.save(save_dir, model_num)


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
