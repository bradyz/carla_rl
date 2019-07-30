import numpy as np

from carla import ColorConverter
from carla import WeatherParameters


PRESET_WEATHERS = {
    1: WeatherParameters.ClearNoon,
    2: WeatherParameters.CloudyNoon,
    3: WeatherParameters.WetNoon,
    4: WeatherParameters.WetCloudyNoon,
    5: WeatherParameters.MidRainyNoon,
    6: WeatherParameters.HardRainNoon,
    7: WeatherParameters.SoftRainNoon,
    8: WeatherParameters.ClearSunset,
    9: WeatherParameters.CloudySunset,
    10: WeatherParameters.WetSunset,
    11: WeatherParameters.WetCloudySunset,
    12: WeatherParameters.MidRainSunset,
    13: WeatherParameters.HardRainSunset,
    14: WeatherParameters.SoftRainSunset,
}

TRAIN_WEATHERS = {
        'clear_noon': WeatherParameters.ClearNoon,                  # 1
        'wet_noon': WeatherParameters.WetNoon,                      # 3
        'hardrain_noon': WeatherParameters.HardRainNoon,            # 6
        'clear_sunset': WeatherParameters.ClearSunset,              # 8
        }

WEATHERS = list(TRAIN_WEATHERS.values())

CROP_SIZE = 192
MAP_SIZE = 320
BACKGROUND = [0, 47, 0]
COLORS = [
        (102, 102, 102),
        (253, 253, 17),
        (204, 6, 5),
        (250, 210, 1),
        (39, 232, 51),
        (0, 0, 142),
        (220, 20, 60)
        ]


TOWNS = ['Town01', 'Town02', 'Town03', 'Town04']
VEHICLE_NAME = 'vehicle.ford.mustang'



def carla_img_to_np(carla_img):
    carla_img.convert(ColorConverter.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype('uint8'))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:,:,:3]
    img = img[:,:,::-1]

    return img


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


def process(observations):
    result = dict()
    result['rgb'] = observations['rgb'].copy()
    result['birdview'] = observations['birdview'].copy()
    result['collided'] = observations['collided']

    control = observations['control']
    control = [control.steer, control.throttle, control.brake]

    result['control'] = np.float32(control)

    measurements = [
            observations['position'],
            observations['orientation'],
            observations['velocity'],
            observations['acceleration'],
            observations['command'].value,
            observations['control'].steer,
            observations['control'].throttle,
            observations['control'].brake,
            observations['control'].manual_gear_shift,
            observations['control'].gear
            ]
    measurements = [x if isinstance(x, np.ndarray) else np.float32([x]) for x in measurements]
    measurements = np.concatenate(measurements, 0)

    result['measurements'] = measurements

    return result


def crop_birdview(birdview, dx=0, dy=0):
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview


def visualize_birdview(birdview):
    """
    0 road
    1 lane
    2 red light
    3 yellow light
    4 green light
    5 vehicle
    6 pedestrian
    """
    h, w = birdview.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND

    for i in range(len(COLORS)):
        canvas[birdview[:,:,i] > 0] = COLORS[i]

    return canvas


def visualize_predicted_birdview(predicted, tau=0.5):
    # mask = np.concatenate([predicted.max(0)[np.newaxis]] * 7, 0)
    # predicted[predicted != mask] = 0
    # predicted[predicted == mask] = 1

    predicted[predicted < tau] = -1

    return visualize_birdview(predicted.transpose(1, 2, 0))


class LocalPlannerNew(object):
    def __init__(self, vehicle, resolution=15, threshold_before=2.5, threshold_after=5.0):
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

        # Max skip avoids misplanning when route includes both lanes.
        self._max_skip = 20
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after

        self._vehicle = vehicle
        self._map = vehicle.get_world().get_map()
        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._map, resolution))
        self._grp.setup()

        self._route = None
        self._waypoints_queue = deque(maxlen=20000)

        self.target = (None, None)
        self.checkpoint = (None, None)
        self.distance_to_goal = float('inf')
        self.distances = deque(maxlen=20000)

    def set_global_plan(self, route):
        self._waypoints_queue.clear()
        self._route = route

        self.distance_to_goal = 0.0

        prev = None

        for node in self._route:
            self._waypoints_queue.append(node)

            cur = node[0].transform.location

            if prev is not None:
                delta = np.sqrt((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2)

                self.distance_to_goal += delta
                self.distances.append(delta)

            prev = cur

        self.target = self._waypoints_queue[0]
        self.checkpoint = (
                self._map.get_waypoint(self._vehicle.get_location()),
                RoadOption.LANEFOLLOW)

    def set_route(self, start, target):
        self.set_global_plan(self._grp.trace_route(start, target))

    def run_step(self):
        assert self._route is not None

        u = self._vehicle.get_transform().location
        max_index = -1

        for i, (node, command) in enumerate(self._waypoints_queue):
            if i > self._max_skip:
                break

            v = node.transform.location
            distance = np.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2)

            if int(self.checkpoint[1]) == 4 and int(command) != 4:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            if distance < threshold:
                self.checkpoint = (node, command)
                max_index = i

        for i in range(max_index + 1):
            if self.distances:
                self.distance_to_goal -= self.distances[0]
                self.distances.popleft()

            self._waypoints_queue.popleft()

        if len(self._waypoints_queue) > 0:
            self.target = self._waypoints_queue[0]

    def calculate_timeout(self, fps=10):
        _numpy = lambda p: np.array([p.transform.location.x, p.transform.location.y])

        distance = 0
        node_prev = None

        for node_cur, _ in self._route:
            if node_prev is None:
                node_prev = node_cur

            distance += np.linalg.norm(_numpy(node_cur) - _numpy(node_prev))
            node_prev = node_cur

        timeout_in_seconds = ((distance / 1000.0) / 5.0) * 3600.0 + 20.0
        timeout_in_frames = timeout_in_seconds * fps

        return timeout_in_frames
