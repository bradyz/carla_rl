import logging

import numpy as np

import gym
from gym import spaces

from .carla_wrapper import CarlaWrapper
from . import carla_utils as cu


logger = logging.getLogger(__name__)


class CarlaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.port = 3000
        self._wrapper = CarlaWrapper(port=self.port)

        self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]), dtype=np.float16)
        self.observation_space = spaces.Box(
                low=0, high=255, shape=(192, 192, 7),
                dtype=np.uint8)

    def step(self, action):
        self._wrapper.apply_control(action)
        self._wrapper.tick()

        obs = self._wrapper.get_observations()
        obs = obs['birdview']
        obs = cu.crop_birdview(obs)

        reward = self._wrapper._tick

        done = self._wrapper.collided

        return obs, reward, done, {}

    def reset(self):
        self._wrapper.init(n_pedestrians=10, n_vehicles=10)
        self._wrapper.ready()

        obs = self._wrapper.get_observations()
        obs = cu.crop_birdview(obs['birdview'])

        return obs

    def render(self, mode='human'):
        tmp = self._wrapper.get_observations()
        tmp = tmp['birdview']
        tmp = cu.crop_birdview(tmp)
        tmp = cu.visualize_birdview(tmp)

        import cv2

        cv2.imshow('asdf', cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
