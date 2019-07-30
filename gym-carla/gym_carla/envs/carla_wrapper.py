import queue
import weakref
import time
import random
import collections

import numpy as np
import carla

from .map_utils import Wrapper as map_utils
from .carla_utils import WEATHERS, TRAIN_WEATHERS


SYNC = False


def get_birdview(observations):
    birdview = [
            observations['road'],
            observations['lane'],
            observations['traffic'],
            observations['vehicle'],
            observations['pedestrian']
            ]
    birdview = [x if x.ndim == 3 else x[...,None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = carla.WorldSettings(
            synchronous_mode=sync,
            no_rendering_mode=True,
            fixed_delta_seconds=0.1)

    world.apply_settings(settings)


class CarlaWrapper(object):
    def __init__(
            self, town='Town01', vehicle_name='vehicle.ford.mustang', port=2000,
            col_threshold=10, big_cam=False, **kwargs):
        self._client = carla.Client('localhost', port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()
        self._vehicle_bp = np.random.choice(self._blueprints.filter(vehicle_name))
        self._vehicle_bp.set_attribute('role_name', 'hero')

        self._tick = 0
        self._player = None

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)

        self._big_cam = big_cam
        self.col_threshold = col_threshold
        self.collided = False
        self._collided_frame_number = -1

        self.invaded = False
        self._invaded_frame_number = -1

        self.n_vehicles = 0
        self.n_pedestrians = 0

        # self._rgb_queue = None
        self.seed = 0

    def spawn_vehicles(self):
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = list()
        blueprints = self._blueprints.filter('vehicle.*')
        spawn_points = self._map.get_spawn_points()

        np.random.seed(self.seed)
        np.random.shuffle(spawn_points)

        for transform in spawn_points[:self.n_vehicles]:
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

        print('Vehicles: %d' % len(self._actor_dict['vehicle']))

    def spawn_pedestrians(self):
        SpawnActor = carla.command.SpawnActor

        np.random.seed(self.seed)

        spawn_points = list()

        for i in range(self.n_pedestrians):
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

        print('Controllers: %d' % len(self._actor_dict['ped_controller']))
        print('Pedestrians: %d' % len(self._actor_dict['pedestrian']))

    def set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        elif weather_string in TRAIN_WEATHERS:
            weather = TRAIN_WEATHERS[weather_string]
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def init(self, start=0, weather='random', n_vehicles=0, n_pedestrians=0):
        if SYNC:
            set_sync_mode(self._client, True)

        self.n_vehicles = n_vehicles or self.n_vehicles
        self.n_pedestrians = n_pedestrians or self.n_pedestrians
        self._start_pose = self._map.get_spawn_points()[start]

        self.clean_up()
        self.spawn_player()
        self._setup_sensors()

        # Hiding away the gore.
        map_utils.init(self._client, self._world, self._map, self._player)

        # Deterministic.
        np.random.seed(self.seed)

        self.set_weather(weather)
        self.spawn_vehicles()
        self.spawn_pedestrians()

        # Busy poll.
        self.ready()

    def spawn_player(self):
        self._player = self._world.spawn_actor(self._vehicle_bp, self._start_pose)
        self._player.set_autopilot(False)

        self._actor_dict['player'].append(self._player)

    def ready(self):
        self.tick()
        self.get_observations()

        for controller in self._actor_dict['ped_controller']:
            controller.start()
            controller.go_to_location(self._world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

        self.tick()
        self.get_observations()

        # with self._rgb_queue.mutex:
            # self._rgb_queue.queue.clear()

        self._time_start = time.time()
        self._tick = 0

    def tick(self):
        tmp = self._world.get_snapshot()

        if self._tick == 0:
            self.start = tmp.timestamp.frame

        self.ticks = tmp.timestamp.frame - self.start

        if SYNC:
            print(self._world.tick())

        self._tick += 1

        # More hiding.
        map_utils.tick()

        # Put here for speed (get() busy polls queue).
        # self.rgb_image = self._rgb_queue.get()

        return True

    def get_observations(self):
        result = dict()
        result.update(map_utils.get_observations())
        result.update({
            # 'rgb': carla_img_to_np(self.rgb_image),
            'birdview': get_birdview(result),
            'collided': self.collided
            })

        return result

    def apply_control(self, control=None):
        if control is not None:
            if not isinstance(control, carla.VehicleControl):
                vehicle_control = carla.VehicleControl()
                vehicle_control.steer = control[0]
                vehicle_control.throttle = control[1]
                vehicle_control.brake = control[2]
                vehicle_control.manual_gear_shift = False
                vehicle_control.hand_brake = False

                control = vehicle_control

            self._player.apply_control(control)

        return {
                't': self._tick,
                'wall': time.time() - self._time_start
                }

    def clean_up(self):
        for controller in self._actor_dict['ped_controller']:
            controller.stop()

        for sensor in self._actor_dict['sensor']:
            sensor.destroy()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()

        self._tick = 0
        self._time_start = time.time()
        self._player = None

        # Clean-up cameras
        # if self._rgb_queue:
            # with self._rgb_queue.mutex:
                # self._rgb_queue.queue.clear()

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        # Camera.
        # self._rgb_queue = queue.Queue()
        # rgb_camera_bp = self._blueprints.find('sensor.camera.rgb')

        # if self._big_cam:
            # rgb_camera_bp.set_attribute('image_size_x', '800')
            # rgb_camera_bp.set_attribute('image_size_y', '600')
            # rgb_camera_bp.set_attribute('fov', '100')
            # rgb_camera = self._world.spawn_actor(
                # rgb_camera_bp,
                # carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15)),
                # attach_to=self._player)
        # else:
            # rgb_camera_bp.set_attribute('image_size_x', '384')
            # rgb_camera_bp.set_attribute('image_size_y', '160')
            # rgb_camera_bp.set_attribute('fov', '90')
            # rgb_camera = self._world.spawn_actor(
                # rgb_camera_bp,
                # carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
                # attach_to=self._player)

        # rgb_camera.listen(self._rgb_queue.put)
        # self._actor_dict['sensor'].append(rgb_camera)

        # Collisions.
        self.collided = False
        self._collided_frame_number = -1

        collision_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.collision'),
                carla.Transform(), attach_to=self._player)
        collision_sensor.listen(
                lambda event: self.__class__._on_collision(weakref.ref(self), event))
        self._actor_dict['sensor'].append(collision_sensor)

        # Lane invasion.
        self.invaded = False
        self._invaded_frame_number = -1

        invasion_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.lane_invasion'),
                carla.Transform(), attach_to=self._player)
        invasion_sensor.listen(
                lambda event: self.__class__._on_invasion(weakref.ref(self), event))
        self._actor_dict['sensor'].append(invasion_sensor)

    @staticmethod
    def _on_collision(weakself, event):
        _self = weakself()

        if not _self:
            return

        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])

        if intensity > _self.col_threshold:
            _self.collided = True
            _self._collided_frame_number = event.frame_number

    @staticmethod
    def _on_invasion(weakself, event):
        _self = weakself()

        if not _self:
            return

        _self.invaded = True
        _self._invaded_frame_number = event.frame_number

    def __del__(self):
        self.clean_up()

        set_sync_mode(self._client, False)

    def render_world(self):
        return map_utils.render_world()

    def world_to_pixel(self, pos):
        return map_utils.world_to_pixel(pos)
