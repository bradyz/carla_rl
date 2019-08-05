import time
import random
import collections

import numpy as np
import gym
import cv2

import carla

from . import carla_utils as cu
from .carla_wrapper import CarlaWrapper
from .map_utils import Wrapper as map_utils


class CarlaMultiEnv(gym.Env):
    def __init__(self, town='Town01', port=3000, n_heroes=2):
        self._client = carla.Client('localhost', port)
        self._client.set_timeout(30.0)

        cu.set_sync_mode(self._client, False)

        self._world = self._client.load_world(town)
        self._map = self._world.get_map()
        self._blueprints = self._world.get_blueprint_library()

        cu.set_sync_mode(self._client, True)

        self._carla_seed = 0
        self._wall_start = time.time()
        self._heroes = list()
        self._actor_dict = collections.defaultdict(list)

        for i in range(n_heroes):
            self._heroes.append(CarlaWrapper(self._client, hero_num=i))

    def step(self, actions):
        self._world.tick()

        observations = self.get_observations()
        rewards = list()
        dones = list()
        infos = list()

        for i, wrapper in enumerate(self._heroes):
            state = wrapper.step(actions[i])

            reward = state.velocity / 5.0 + -int(wrapper._state.collided)
            done = wrapper._state.collided
            info = wrapper._state.tick

            if done:
                wrapper.reset()

            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return observations, rewards, dones, infos

    def get_observations(self):
        observations = list()

        for wrapper in self._heroes:
            map_utils.tick()

            observation = map_utils.get_crop(wrapper._player.get_transform())
            observation = cu.crop_birdview(observation)
            # observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_NEAREST)
            observations.append(observation)

        return observations

    def reset(self, weather='random', n_vehicles=0, n_pedestrians=0):
        self._clean_up()

        map_utils.init(self._world)

        for wrapper in self._heroes:
            wrapper.reset()

        # Deterministic.
        np.random.seed(self._carla_seed)

        self._set_weather(weather)
        self._spawn_vehicles(n_vehicles)
        self._spawn_pedestrians(n_pedestrians)

        print('Vehicles: %d' % len(self._actor_dict['vehicle']))
        print('Controllers: %d' % len(self._actor_dict['ped_controller']))
        print('Pedestrians: %d' % len(self._actor_dict['pedestrian']))

        return self.get_observations()

    def render(self):
        birdviews = list()

        for i, birdview in enumerate(self.get_observations()):
            birdview = cu.visualize_birdview(birdview)
            birdview = cv2.cvtColor(birdview, cv2.COLOR_RGB2BGR)
            birdviews.append(birdview)

        for i, birdview in enumerate(birdviews):
            cv2.imshow(str(i), birdview)
            cv2.waitKey(1)

    def _set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(cu.WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def _spawn_vehicles(self, n_vehicles):
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = list()
        blueprints = self._blueprints.filter('vehicle.*')
        spawn_points = self._map.get_spawn_points()

        np.random.seed(self._carla_seed)
        np.random.shuffle(spawn_points)

        for transform in spawn_points[:n_vehicles]:
            blueprint = random.choice(blueprints)
            blueprint.set_attribute('role_name', 'autopilot')

            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        vehicles = list()

        for response in self._client.apply_batch_sync(batch):
            if not response.error:
                vehicles.append(response.actor_id)

        self._actor_dict['vehicle'].extend(self._world.get_actors(vehicles))

    def _spawn_pedestrians(self, n_pedestrians):
        SpawnActor = carla.command.SpawnActor

        np.random.seed(self._carla_seed)

        spawn_points = list()

        for i in range(n_pedestrians):
            spawn_point = carla.Transform()
            loc = self._world.get_random_location_from_navigation()

            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        blueprints = self._blueprints.filter('walker.pedestrian.*')
        batch = list()

        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)

            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            batch.append(SpawnActor(walker_bp, spawn_point))

        walkers = list()
        controllers = list()

        for result in self._client.apply_batch_sync(batch, True):
            if not result.error:
                walkers.append(result.actor_id)

        walker_controller_bp = self._blueprints.find('controller.ai.walker')
        batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker) for walker in walkers]

        for result in self._client.apply_batch_sync(batch, True):
            if not result.error:
                controllers.append(result.actor_id)

        self._actor_dict['ped_controller'].extend(self._world.get_actors(controllers))
        self._actor_dict['pedestrian'].extend(self._world.get_actors(walkers))

        for controller in self._actor_dict['ped_controller']:
            controller.start()
            controller.go_to_location(self._world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

    def _clean_up(self):
        for controller in self._actor_dict['ped_controller']:
            controller.stop()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()
        self._wall_start = time.time()

    def __del__(self):
        self._clean_up()

        cu.set_sync_mode(self._client, False)
