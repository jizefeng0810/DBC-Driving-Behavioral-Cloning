import collections
import queue
import weakref
import time
import random
import math
import logging
import numpy as np
import carla

from .map_utils import Wrapper as map_utils


MAX_SPEED_MS = 15
SPAWNS = [
        304,303,302,301,94,93,92,91,90,89,88,87,
        352,351,350,60,61,62,63,313,314,315]


PRESET_WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,   #
    2: carla.WeatherParameters.CloudyNoon,
    3: carla.WeatherParameters.WetNoon,
    4: carla.WeatherParameters.WetCloudyNoon,
    5: carla.WeatherParameters.MidRainyNoon,
    6: carla.WeatherParameters.HardRainNoon,    #
    7: carla.WeatherParameters.SoftRainNoon,
    8: carla.WeatherParameters.ClearSunset,     #
    9: carla.WeatherParameters.CloudySunset,
    10: carla.WeatherParameters.WetSunset,
    11: carla.WeatherParameters.WetCloudySunset,
    12: carla.WeatherParameters.MidRainSunset,
    13: carla.WeatherParameters.HardRainSunset,
    14: carla.WeatherParameters.SoftRainSunset,
}

WEATHERS = list(PRESET_WEATHERS.values())
VEHICLE_NAME = '*mustang*'
COLLISION_THRESHOLD = 10


def _carla_img_to_numpy(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    return array


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 0.1

    world.apply_settings(settings)


class CarlaEnv(object):
    def __init__(self, town='Town04', port=2000, **kwargs):
        self._client = carla.Client('localhost', port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        print(town)
        self._world = self._client.load_world(town)
        self._world.set_weather(PRESET_WEATHERS[8])     # 设置天气 1：clear 6：hard rain 8：sunset
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0
        self._player = None

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)

        self.collided = False
        self._collided_frame_number = -1

        self._rgb_queue = None
        self._seg_queue = None

        self.tm = self._client.get_trafficmanager(8000)

    def _spawn_vehicles(self, n_vehicles, n_retries=30):
        spawn_points = self._map.get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        synchronous_master = False

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        vehicles_list = []
        vehicles_num = 0
        for n, transform in enumerate(spawn_points):
            if random.random() < 0.2:
                if vehicles_num >= min(number_of_spawn_points, n_vehicles):
                    break
                vehicles_num += 1
                # blueprint = random.choice(self.blueprints)
                blueprint = random.choice(self._blueprints.filter('vehicle.nissan.micra'))
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self.tm.get_port())))

        for response in self._client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

    def _spawn_pedestrians(self, n_pedestrians):
        SpawnActor = carla.command.SpawnActor

        peds_spawned = 0

        walkers = []

        while peds_spawned < n_pedestrians:
            spawn_points = []
            _walkers = []

            for i in range(n_pedestrians - peds_spawned):
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()

                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            blueprints = self._blueprints.filter('walker.pedestrian.*')
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprints)

                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')

                batch.append(SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(batch, True):
                if not result.error:
                    peds_spawned += 1
                    _walkers.append(result.actor_id)

            walkers.extend(_walkers)

        self._actor_dict['pedestrian'].extend(self._world.get_actors(walkers))

    def _set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def reset(self, start=0, weather='random', n_vehicles=10, n_pedestrians=10, seed=0):
        is_ready = False

        while not is_ready:
            np.random.seed(seed)

            self._clean_up()
            # self._spawn_player(self._map.get_spawn_points()[np.random.choice(SPAWNS)])
            self._spawn_player(np.random.choice(self._map.get_spawn_points()))
            self._setup_sensors()

            # self._set_weather(weather)
            # self._spawn_pedestrians(n_pedestrians)
            self._spawn_vehicles(n_vehicles)

            print('%d / %d vehicles spawned.' % (len(self._actor_dict['vehicle']), n_vehicles))
            # print('%d / %d pedestrians spawned.' % (len(self._actor_dict['pedestrian']), n_pedestrians))

            is_ready = self.ready()

    def _spawn_player(self, start_pose):
        vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE_NAME))
        vehicle_bp.set_attribute('role_name', 'hero')

        # ----------------------------------------------#
        # route 0
        # A
        start_pose.location.x = 110
        start_pose.location.y = 228
        start_pose.rotation.yaw = -18

        # B
        start_pose.location.x = -37.1
        start_pose.location.y = -229.1
        start_pose.rotation.yaw = 131

        # C
        start_pose.location.x = -486
        start_pose.location.y = 37
        start_pose.rotation.yaw = 131

        # route 1
        # start_pose.location.x = 82
        # start_pose.location.y = -2
        # start_pose.rotation.yaw = 180

        # test driving stability
        start_pose.location.x, start_pose.location.y, start_pose.rotation.yaw = 401.6,-38.4,-90
        start_pose.location.x, start_pose.location.y, start_pose.rotation.yaw = -505.2, 228.5, 90
        start_pose.location.x, start_pose.location.y, start_pose.rotation.yaw = -412.2, 16.2, 180

        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        # ----------------------------------------------#

        self._player = self._world.spawn_actor(vehicle_bp, start_pose)

        map_utils.init(self._player)

        self._actor_dict['player'].append(self._player)

    def ready(self, ticks=10):
        self.step()

        for _ in range(ticks):
            self.step()

        with self._rgb_queue.mutex:
            self._rgb_queue.queue.clear()

        with self._seg_queue.mutex:
            self._seg_queue.queue.clear()

        self._time_start = time.time()
        self._tick = 0

        return not self.collided

    def step(self, control=None):
        if control is not None:
            self._player.apply_control(control) # 转向角、油门、刹车控制

        for i, vehicle in enumerate(self._actor_dict['vehicle']):   # 限速
            if self._tick % 200 == 0:
                self._vehicle_speeds[i] = np.random.randint(MAX_SPEED_MS // 2, MAX_SPEED_MS + 1)

            max_speed = self._vehicle_speeds[i]
            velocity = vehicle.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])    # 合速度

            if speed > max_speed:
                vehicle.set_velocity(vehicle.get_velocity() * 0.9)

        self._world.tick()
        map_utils.tick()

        self._tick += 1

        # Put here for speed (get() busy polls queue).
        rgb = None
        while rgb is None or self._rgb_queue.qsize() > 0:
            rgb = self._rgb_queue.get()

        seg = None
        while seg is None or self._seg_queue.qsize() > 0:
            seg = self._seg_queue.get()
        vel = self._player.get_velocity()
        control = self._player.get_control()

        result = map_utils.get_observations()   # add the result of birdview and velocity
        result.update({
            'collided': self.collided,
            'rgb': _carla_img_to_numpy(rgb),
            'segmentation': _carla_img_to_numpy(seg)[:, :, 0],
            'speed': 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2),
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
            })
        return result

    def _clean_up(self):
        for vehicle in self._actor_dict['vehicle']:
            vehicle.destroy()

        for sensor in self._actor_dict['sensor']:
            sensor.destroy()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()
        self._vehicle_speeds = list()

        self._tick = 0
        self._time_start = time.time()

        self._player = None

        # Clean-up cameras
        if self._rgb_queue:
            with self._rgb_queue.mutex:
                self._rgb_queue.queue.clear()

        if self._seg_queue:
            with self._seg_queue.mutex:
                self._seg_queue.queue.clear()

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        # Camera.
        self._rgb_queue = queue.Queue()

        rgb_camera_bp = self._blueprints.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '384')
        rgb_camera_bp.set_attribute('image_size_y', '160')
        rgb_camera_bp.set_attribute('fov', '90')
        rgb_camera = self._world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self._player)
        rgb_camera.listen(self._rgb_queue.put)

        self._actor_dict['sensor'].append(rgb_camera)

        self._seg_queue = queue.Queue()

        seg_camera_bp = self._blueprints.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '384')
        seg_camera_bp.set_attribute('image_size_y', '160')
        seg_camera_bp.set_attribute('fov', '90')

        seg_camera = self._world.spawn_actor(
            seg_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self._player)
        seg_camera.listen(self._seg_queue.put)

        self._actor_dict['sensor'].append(seg_camera)

        # Collisions.
        self.collided = False
        self._collided_frame_number = -1

        collision_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.collision'),
                carla.Transform(), attach_to=self._player)
        collision_sensor.listen(lambda event: self.__class__._on_collision(weakref.ref(self), event))

        self._actor_dict['sensor'].append(collision_sensor)

    @staticmethod
    def _on_collision(weakself, event):
        _self = weakself()

        if not _self:
            return

        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])

        print(intensity)

        if intensity > COLLISION_THRESHOLD:
            _self.collided = True
            _self._collided_frame_number = event.frame_number

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._clean_up()

        set_sync_mode(self._client, False)
