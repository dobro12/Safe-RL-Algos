import safety_gymnasium as safety_gym
import gymnasium as gym
import numpy as np

class SafetyGymEnv(gym.Env):
    def __init__(self, env_name, **args):
        self._env = safety_gym.make(env_name, **args)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_space = gym.spaces.box.Box(
            -np.inf*np.ones(1, dtype=np.float64), 
            np.inf*np.ones(1, dtype=np.float64), 
            dtype=np.float64,
        )
        self.cost_space = gym.spaces.box.Box(
            -np.inf*np.ones(1, dtype=np.float64), 
            np.inf*np.ones(1, dtype=np.float64), 
            dtype=np.float64,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, cost, terminate, truncate, info = self._env.step(action)
        reward = np.array([reward, cost])
        return obs, reward, terminate, truncate, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()
