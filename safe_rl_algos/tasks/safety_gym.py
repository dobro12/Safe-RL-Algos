import safety_gymnasium as safety_gym
import gymnasium as gym
import numpy as np

class SafetyGymEnv(gym.Env):
    def __init__(self, env_name, **args):
        self.env_name = env_name
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

    ################
    # public methods
    ################

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, cost, terminate, truncate, info = self._env.step(action)
        reward = np.array([reward, self._cost(cost)])
        return obs, reward, terminate, truncate, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()

    #################
    # private methods
    #################

    def _cost(self, original_cost, h_coeff=10.0):
        if not "goal" in self.name.lower():
            return original_cost

        hazard_pos_list = None
        hazard_size = None
        hazard_dist = np.inf
        for obs in self._env.task._obstacles:
            if obs.name == "hazards":
                hazard_pos_list = obs.pos
                hazard_size = obs.size
                break
        assert hazard_pos_list is not None
        for hazard_pos in hazard_pos_list:
            dist = self._env.task.agent.dist_xy(hazard_pos)
            if dist < hazard_dist:
                hazard_dist = dist
        cost = 1.0/(1.0 + np.exp((hazard_dist - hazard_size)*h_coeff))
        return cost