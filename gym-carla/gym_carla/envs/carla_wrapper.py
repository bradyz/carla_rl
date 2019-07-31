import weakref
import time
import collections

import numpy as np
import carla


COLLISION_THRESHOLD = 10
VEHICLE = 'vehicle.ford.mustang'


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = carla.WorldSettings(
            synchronous_mode=sync,
            no_rendering_mode=True,
            fixed_delta_seconds=0.1)

    world.apply_settings(settings)


class CarlaState(object):
    def __init__(self):
        self.tick = 0

        self.collided = False
        self.collided_frame_number = -1

        self.invaded = False
        self.invaded_frame_number = -1


class CarlaWrapper(object):
    def __init__(self, client, hero_num=0):
        self._client = client
        self._world = self._client.get_world()
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()
        self._vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE))
        self._vehicle_bp.set_attribute('role_name', 'hero_%s' % hero_num)

        self._player = None
        self._actor_dict = collections.defaultdict(list)

        self._state = CarlaState()

    def reset(self, start=None):
        self._clean_up()

        for _ in range(10):
            try:
                start = np.random.randint(len(self._map.get_spawn_points()))
                start_pose = self._map.get_spawn_points()[start]

                self._player = self._world.spawn_actor(self._vehicle_bp, start_pose)
                self._player.set_autopilot(False)

                break
            except RuntimeError:
                pass

        self._actor_dict['player'].append(self._player)
        self._setup_sensors()

    def tick(self):
        snapshot = self._world.get_snapshot()

        if self._tick == 0:
            self.start = snapshot.timestamp.frame

        self.ticks = snapshot.timestamp.frame - self.start

        self._tick += 1

        return True

    def get_hero_measurements(self):
        pos = self.hero_actor.get_location()
        ori = self.hero_actor.get_transform().get_forward_vector()
        vel = self.hero_actor.get_velocity()
        acc = self.hero_actor.get_acceleration()

        return {
                'position': np.float32([pos.x, pos.y, pos.z]),
                'orientation': np.float32([ori.x, ori.y]),
                'velocity': np.float32([vel.x, vel.y, vel.z]),
                'acceleration': np.float32([acc.x, acc.y, acc.z])
                }

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

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        # Collisions.
        collision_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.collision'),
                carla.Transform(), attach_to=self._player)
        collision_sensor.listen(lambda event: self.__class__._on_collision(weakref.ref(self), event))
        self._actor_dict['sensor'].append(collision_sensor)

        # Lane invasion.
        invasion_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.lane_invasion'),
                carla.Transform(), attach_to=self._player)
        invasion_sensor.listen(lambda event: self.__class__._on_invasion(weakref.ref(self), event))
        self._actor_dict['sensor'].append(invasion_sensor)

    @staticmethod
    def _on_collision(weakself, event):
        _self = weakself()

        if not _self:
            return

        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])

        if intensity > COLLISION_THRESHOLD:
            _self._state.collided = True
            _self._state.collided_frame_number = event.frame_number

    @staticmethod
    def _on_invasion(weakself, event):
        _self = weakself()

        if not _self:
            return

        _self._state.invaded = True
        _self._state.invaded_frame_number = event.frame_number

    def _clean_up(self):
        for controller in self._actor_dict['ped_controller']:
            controller.stop()

        for sensor in self._actor_dict['sensor']:
            sensor.destroy()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._player = None
        self._actor_dict.clear()
        self._state = CarlaState()

        self._tick = 0
        self._time_start = time.time()

    def __del__(self):
        self._clean_up()
