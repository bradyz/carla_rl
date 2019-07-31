import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Carla-v0',
    entry_point='gym_carla.envs:CarlaMultiEnv',
)

register(
    id='CarlaPointGoal-v0',
    entry_point='gym_carla.envs:CarlaPointGoalEnv',
)
