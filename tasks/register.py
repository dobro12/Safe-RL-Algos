from safety_gym.envs.suite import SafexpEnvBase
from gym.envs.registration import register

# ========= locomotion tasks ========= #
register(
    id='Safexp-PointGoal3-v0',
    entry_point='tasks.safety_gym:PointGoal3Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Safexp-CarGoal3-v0',
    entry_point='tasks.safety_gym:CarGoal3Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)
# ==================================== #
