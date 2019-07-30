import os

import numpy as np
import pygame

import carla


# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================
COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

COLOR_TRAFFIC_RED = pygame.Color(255, 0, 0)
COLOR_TRAFFIC_YELLOW = pygame.Color(0, 255, 0)
COLOR_TRAFFIC_GREEN = pygame.Color(0, 0, 255)

# Module Defines
MODULE_WORLD = 'WORLD'

PIXELS_PER_METER = 5
PIXELS_AHEAD_VEHICLE = 100


def surface_to_numpy(surface):
    return np.swapaxes(pygame.surfarray.array3d(surface), 0, 1)


# ==============================================================================
# -- ModuleManager -------------------------------------------------------------
# ==============================================================================
class ModuleManager(object):
    def __init__(self):
        self.modules = []

    def register_module(self, module):
        self.modules.append(module)

    def clear_modules(self):
        del self.modules[:]

    def tick(self, clock):
        for module in self.modules:
            module.tick(clock)

    def render(self, display):
        display.fill(COLOR_ALUMINIUM_4)

        for module in self.modules:
            module.render(display)

    def get_module(self, name):
        for module in self.modules:
            if module.name == name:
                return module

    def start_modules(self):
        for module in self.modules:
            module.start()


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter=10):
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.big_lane_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.draw_road_map(
                self.big_map_surface, self.big_lane_surface,
                carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)
        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface

    def draw_road_map(self, map_surface, lane_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        # map_surface.fill(COLOR_ALUMINIUM_4)
        map_surface.fill(COLOR_BLACK)
        precision = 0.05

        def draw_lane_marking(surface, points, solid=True):
            if solid:
                # pygame.draw.lines(surface, COLOR_ORANGE_0, False, points, 2)
                pygame.draw.lines(surface, COLOR_WHITE, False, points, 2)
            else:
                broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
                for line in broken_lines:
                    # pygame.draw.lines(surface, COLOR_ORANGE_0, False, line, 2)
                    pygame.draw.lines(surface, COLOR_WHITE, False, line, 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            start = transform.location
            end = start + 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        start, end]], 4)
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        left, start, right]], 4)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            w = carla_map.get_waypoint(lateral_shift(waypoint.transform, shift), project_to_road=False)
            if w is None or w.road_id != waypoint.road_id:
                return True
            else:
                return (w.lane_id * waypoint.lane_id < 0) or w.lane_id == waypoint.lane_id

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)

        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            polygon = left_marking + [x for x in reversed(right_marking)]
            polygon = [world_to_pixel(x) for x in polygon]

            if len(polygon) > 2:
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon, 10)
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon)

            if not waypoint.is_intersection:
                sample = waypoints[int(len(waypoints) / 2)]
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in left_marking],
                    does_cross_solid_line(sample, -sample.lane_width * 1.1))
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in right_marking],
                    does_cross_solid_line(sample, sample.lane_width * 1.1))

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))


class ModuleWorld(object):
    def __init__(self, name, client, world, town_map, hero_actor):
        self.name = name

        # World data
        self.client = client
        self.world = world
        self.town_map = town_map
        self.actors_with_transforms = []
        self.surface_size = [0, 0]
        # Hero actor
        self.hero_actor = hero_actor
        self.hero_transform = hero_actor.get_transform()

        self.scale_offset = [0, 0]

        # Map info
        self.map_image = None
        self.original_surface_size = None

        self.self_surface = None
        self.vehicle_surface = None
        self.walker_surface = None

        self.hero_map_surface = None
        self.hero_lane_surface = None
        self.hero_vehicle_surface = None
        self.hero_walker_surface = None
        self.hero_traffic_light_surface = None

        self.window_map_surface = None
        self.window_lane_surface = None
        self.window_vehicle_surface = None
        self.window_walker_surface = None
        self.window_traffic_light_surface = None

        self.hero_map_image = None
        self.hero_lane_image = None
        self.hero_vehicle_image = None
        self.hero_walker_image = None
        self.hero_traffic_image = None

    def get_rendered_surfaces(self):
        return (
            self.hero_map_image,
            self.hero_lane_image,
            self.hero_vehicle_image,
            self.hero_walker_image,
            self.hero_traffic_image,
        )

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

    def start(self):
        # Create Surfaces
        self.map_image = MapImage(self.world, self.town_map, PIXELS_PER_METER)

        self.original_surface_size = 320
        self.surface_size = self.map_image.big_map_surface.get_width()
        self.window_width = 320
        self.window_height = 320

        map_width = self.map_image.map_surface.get_width()
        map_height = self.map_image.map_surface.get_height()

        # Render Actors
        self.vehicle_surface = pygame.Surface((map_width, map_height))
        self.vehicle_surface.set_colorkey(COLOR_BLACK)
        self.self_surface = pygame.Surface((map_width, map_height))
        self.self_surface.set_colorkey(COLOR_BLACK)
        self.walker_surface = pygame.Surface((map_width, map_height))
        self.walker_surface.set_colorkey(COLOR_BLACK)
        self.traffic_light_surface = pygame.Surface((map_width, map_height))
        self.traffic_light_surface.set_colorkey(COLOR_BLACK)

        scaled_size = self.original_surface_size * (1.0 / 1.0)

        self.hero_map_surface = pygame.Surface((scaled_size, scaled_size)).convert()
        self.hero_lane_surface = pygame.Surface((scaled_size, scaled_size)).convert()
        self.hero_vehicle_surface = pygame.Surface((scaled_size, scaled_size)).convert()
        self.hero_walker_surface = pygame.Surface((scaled_size, scaled_size)).convert()
        self.hero_traffic_light_surface = pygame.Surface((scaled_size, scaled_size)).convert()

        self.window_map_surface = pygame.Surface((self.window_width, self.window_height)).convert()
        self.window_lane_surface = pygame.Surface((self.window_width, self.window_height)).convert()
        self.window_vehicle_surface = pygame.Surface((self.window_width, self.window_height)).convert()
        self.window_walker_surface = pygame.Surface((self.window_width, self.window_height)).convert()
        self.window_traffic_light_surface = pygame.Surface((self.window_width, self.window_height)).convert()

    def tick(self, clock):
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]

        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()

    def _split_actors(self):
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)

        return vehicles, traffic_lights, speed_limits, walkers

    def get_bounding_box(self, actor):
        bb = actor.trigger_volume.extent
        corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners

    def _render_traffic_lights(self, surface, list_tl, world_to_pixel, world_to_pixel_width):
        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)

            if tl.state == carla.TrafficLightState.Red:
                color = COLOR_TRAFFIC_RED
            elif tl.state == carla.TrafficLightState.Yellow:
                color = COLOR_TRAFFIC_YELLOW
            elif tl.state == carla.TrafficLightState.Green:
                color = COLOR_TRAFFIC_GREEN
            else:
                continue

            pygame.draw.circle(surface, color, pos, world_to_pixel_width(1.0))

    def _render_walkers(self, surface, list_w, world_to_pixel):
        for w in list_w:
            color = COLOR_WHITE

            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y)
            ]
            w[1].transform(corners)

            corners = [world_to_pixel(p) for p in corners]
            # print (corners)
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, vehicle_surface, self_surface, list_v, world_to_pixel):
        for v in list_v:
            color = COLOR_WHITE

            if v[0].attributes['role_name'] == 'hero':
                surface = self_surface
                surface = vehicle_surface
            else:
                surface = vehicle_surface

            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [
                    carla.Location(x=-bb.x, y=-bb.y),
                    carla.Location(x=-bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=-bb.y)
                   ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def render_actors(
            self, vehicle_surface, self_surface, walker_surface,
            traffic_light_surface, vehicles, traffic_lights,
            speed_limits, walkers):
        # Static actors
        self._render_traffic_lights(
                traffic_light_surface,
                [tl[0] for tl in traffic_lights],
                self.map_image.world_to_pixel,
                self.map_image.world_to_pixel_width)

        # Dynamic actors
        self._render_vehicles(
                vehicle_surface, self_surface, vehicles,
                self.map_image.world_to_pixel)
        self._render_walkers(
                walker_surface, walkers,
                self.map_image.world_to_pixel)

    def clip_surfaces(self, clipping_rect):
        self.vehicle_surface.set_clip(clipping_rect)
        self.walker_surface.set_clip(clipping_rect)
        self.traffic_light_surface.set_clip(clipping_rect)

    def render(self, display):
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()

        self.vehicle_surface.fill(COLOR_BLACK)
        self.walker_surface.fill(COLOR_BLACK)
        self.traffic_light_surface.fill(COLOR_BLACK)

        self.render_actors(
            self.vehicle_surface, self.self_surface, self.walker_surface,
            self.traffic_light_surface, vehicles, traffic_lights,
            speed_limits, walkers)

        hero_front = self.hero_transform.get_forward_vector()
        hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)

        clip = [0, 0]
        clip[0] += hero_location_screen[0] - self.hero_map_surface.get_width() / 2
        clip[0] += hero_front.x * PIXELS_AHEAD_VEHICLE
        clip[1] += hero_location_screen[1] - self.hero_map_surface.get_height() / 2
        clip[1] += hero_front.y * PIXELS_AHEAD_VEHICLE

        # Apply clipping rect
        clipping_rect = pygame.Rect(
                clip[0], clip[1],
                self.hero_map_surface.get_width(),
                self.hero_map_surface.get_height())

        self.clip_surfaces(clipping_rect)

        self.hero_map_surface.fill(COLOR_BLACK)
        self.hero_vehicle_surface.fill(COLOR_BLACK)
        self.hero_walker_surface.fill(COLOR_BLACK)
        self.hero_traffic_light_surface.fill(COLOR_BLACK)

        self.hero_map_surface.blit(self.map_image.map_surface, (-clip[0], -clip[1]))
        self.hero_lane_surface.blit(self.map_image.lane_surface, (-clip[0], -clip[1]))
        self.hero_vehicle_surface.blit(self.vehicle_surface, (-clip[0], -clip[1]))
        self.hero_walker_surface.blit(self.walker_surface, (-clip[0], -clip[1]))
        self.hero_traffic_light_surface.blit(self.traffic_light_surface, (-clip[0], -clip[1]))

        rz = pygame.transform.rotozoom
        angle = self.hero_transform.rotation.yaw + 90
        scale = 1.0

        rotated_map_surface = rz(self.hero_map_surface, angle, scale)
        rotated_lane_surface = rz(self.hero_lane_surface, angle, scale)
        rotated_vehicle_surface = rz(self.hero_vehicle_surface, angle, scale)
        rotated_walker_surface = rz(self.hero_walker_surface, angle, scale)
        rotated_traffic_surface = rz(self.hero_traffic_light_surface, angle, scale)

        center = (display.get_width() / 2, display.get_height() / 2)
        rotation_map_pivot = rotated_map_surface.get_rect(center=center)
        rotation_lane_pivot = rotated_lane_surface.get_rect(center=center)
        rotation_vehicle_pivot = rotated_vehicle_surface.get_rect(center=center)
        rotation_walker_pivot = rotated_walker_surface.get_rect(center=center)
        rotation_traffic_pivot = rotated_traffic_surface.get_rect(center=center)

        self.window_map_surface.blit(rotated_map_surface, rotation_map_pivot)
        self.window_lane_surface.blit(rotated_lane_surface, rotation_lane_pivot)
        self.window_vehicle_surface.blit(rotated_vehicle_surface, rotation_vehicle_pivot)
        self.window_walker_surface.blit(rotated_walker_surface, rotation_walker_pivot)
        self.window_traffic_light_surface.blit(rotated_traffic_surface, rotation_traffic_pivot)

        # Save surface as rgb array
        self.hero_map_image = surface_to_numpy(self.window_map_surface)[:,:,0]
        self.hero_lane_image = surface_to_numpy(self.window_lane_surface)[:,:,0]
        self.hero_vehicle_image = surface_to_numpy(self.window_vehicle_surface)[:,:,0]
        self.hero_walker_image = surface_to_numpy(self.window_walker_surface)[:,:,0]
        self.hero_traffic_image = surface_to_numpy(self.window_traffic_light_surface)


class CarlaSurface(object):
    def __init__(self, name):
        self.name = name



# ==============================================================================
# -- Global Objects ------------------------------------------------------------
# ==============================================================================
module_manager = ModuleManager()

# ==============================================================================
# bradyz: Wrap all this --------------------------------------------------------
# ==============================================================================
class Wrapper(object):
    clock = None
    display = None
    world_module = None

    @classmethod
    def init(cls, client, world, carla_map, player):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        module_manager.clear_modules()

        pygame.init()
        display = pygame.display.set_mode((320, 320), 0, 32)
        pygame.display.flip()

        # Set map drawer module
        world_module = ModuleWorld(MODULE_WORLD, client, world, carla_map, player)

        # Register Modules
        module_manager.register_module(world_module)
        module_manager.start_modules()

        cls.world_module = world_module
        cls.display = display
        cls.clock = pygame.time.Clock()

    @classmethod
    def tick(cls):
        module_manager.tick(cls.clock)
        module_manager.render(cls.display)

    @classmethod
    def get_observations(cls):
        road, lane, vehicle, pedestrian, traffic = cls.world_module.get_rendered_surfaces()

        result = cls.world_module.get_hero_measurements()
        result.update({
                'road': np.uint8(road),
                'lane': np.uint8(lane),
                'vehicle': np.uint8(vehicle),
                'pedestrian': np.uint8(pedestrian),
                'traffic': np.uint8(traffic),
                })

        pygame.display.flip()

        return result

    @staticmethod
    def clear():
        module_manager.clear_modules()

    @classmethod
    def render_world(cls):
        map_surface = cls.world_module.map_image.big_map_surface
        map_image = np.swapaxes(pygame.surfarray.array3d(map_surface), 0, 1)

        return map_image

    @classmethod
    def world_to_pixel(cls, pos):
        return cls.world_module.map_image.world_to_pixel(pos)
