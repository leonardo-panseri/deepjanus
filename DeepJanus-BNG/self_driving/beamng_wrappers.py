from collections import namedtuple

import numpy as np
from PIL import Image
from beamngpy import angle_to_quat, Vehicle, Scenario, BeamNGpy
from beamngpy.sensors import GForces, Electrics, Damage, Timer, Camera
from beamngpy.vehicle import Sensors

from self_driving.beamng_tig_maps import maps
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.decal_road import DecalRoad
from self_driving.utils import get_node_coords


class BeamNGPose:
    """Represents a combination of position and rotation for an object in the simulator world"""

    def __init__(self, pos=None, rot=None):
        self.pos = pos if pos else (0, 0, 0)
        self.rot = rot if rot else (0, 0, 0)

    def rot_quaternion(self):
        """Converts the rotation from euler angles to a quaternion."""
        return angle_to_quat(self.rot)


# Type containing all vehicle information
VehicleState = namedtuple('VehicleState',
                          ['timer', 'damage', 'pos', 'dir', 'vel', 'gforces', 'gforces2', 'steering',
                           'steering_input', 'brake', 'brake_input', 'throttle', 'throttle_input', 'throttleFactor',
                           'engineThrottle', 'wheelspeed', 'vel_kmh'])


class BeamNGVehicle:
    """Encapsulates a BeamNGpy vehicle, providing all necessary methods to interact with it and retrieve its state"""

    def __init__(self, cameras=False):
        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='TIG', color='Red')
        self.vehicle.attach_sensor('gforces', GForces())
        self.vehicle.attach_sensor('electrics', Electrics())
        self.vehicle.attach_sensor('damage', Damage())
        self.vehicle.attach_sensor('timer', Timer())

        self.start_pose: BeamNGPose = BeamNGPose()

        self.cameras: BeamNGVehicleCameras | None = None
        if cameras:
            self.cameras = BeamNGVehicleCameras(self.vehicle)

        self.sensors: Sensors | None = None
        self.state: VehicleState | None = None

    def add_to_scenario(self, scenario: Scenario):
        """Adds the vehicle to a scenario."""
        scenario.add_vehicle(self.vehicle, pos=self.start_pose.pos,
                             rot_quat=self.start_pose.rot_quaternion())

    def setup_cameras(self, bng: BeamNGpy):
        """Sets up vehicle-mounted cameras. If this vehicle was instantiated with flag camera=False, this method
        does nothing."""
        if self.cameras:
            self.cameras.setup_cameras(bng)

    def control(self, steering: float, throttle: float, brake=.0):
        """Sends commands to the simulated vehicle."""
        self.vehicle.control(steering, throttle, brake)

    def get_state(self) -> VehicleState:
        """Returns the last recorded vehicle state. Call update_state() first to ensure that data is up-to-date."""
        return self.state

    def get_vehicle_bbox(self) -> dict:
        """Gets the bounding box of the vehicle."""
        return self.vehicle.get_bbox()

    def update_state(self):
        """Polls sensors and updates the state of the vehicle."""
        self.vehicle.poll_sensors()
        self.sensors = self.vehicle.sensors

        ele = self.sensors['electrics']
        gforces = self.sensors['gforces']

        vel = tuple(self.vehicle.state['vel'])

        self.state = VehicleState(timer=self.sensors['timer']['time'],
                                  damage=self.sensors['damage']['damage'],
                                  pos=tuple(self.vehicle.state['pos']),
                                  dir=tuple(self.vehicle.state['dir']),
                                  vel=vel,
                                  gforces=(gforces['gx'], gforces['gy'], gforces['gz']),
                                  gforces2=(gforces['gx2'], gforces['gy2'], gforces['gz2']),
                                  steering=ele.get('steering', None),
                                  steering_input=ele.get('steering_input', None),
                                  brake=ele.get('brake', None),
                                  brake_input=ele.get('brake_input', None),
                                  throttle=ele.get('throttle', None),
                                  throttle_input=ele.get('throttle_input', None),
                                  throttleFactor=ele.get('throttleFactor', None),
                                  engineThrottle=ele.get('engineThrottle', None),
                                  wheelspeed=ele.get('wheelspeed', None),
                                  vel_kmh=int(round(np.linalg.norm(vel) * 3.6)))


class BeamNGVehicleCameras:
    """Encapsulates cameras mounted on a vehicle, providing all necessary methods to interact with them"""

    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle

        self.cam_center: Camera | None = None
        # self.cam_left: Camera | None = None
        # self.cam_right: Camera | None = None

    def setup_cameras(self, bng: BeamNGpy, direction=(0, -1, 0), fov=120, resolution=(320, 160), y=-2.2, z=1.0):
        """Creates cameras in the simulation. Note that the simulation must be up and running for this to
        succeed."""
        self.cam_center = Camera('cam_center', bng, self.vehicle, pos=(-0.3, y, z),
                                 dir=direction, field_of_view_y=fov, resolution=resolution,
                                 requested_update_time=0.1, is_using_shared_memory=True, is_render_annotations=False,
                                 is_render_depth=False, is_streaming=True)
        # self.cam_left = Camera('cam_left', self.bng, self.vehicle, pos=(-1.3, y, z),
        #                        dir=direction, field_of_view_y=fov, resolution=resolution,
        #                        requested_update_time=0.1)
        # self.cam_right = Camera('cam_right', self.bng, self.vehicle, pos=(0.4, self.y, self.z),
        #                         dir=direction, field_of_view_y=fov, resolution=resolution,
        #                         requested_update_time=0.1)

    def capture_image_center(self) -> Image:
        """Captures an image from the central camera."""
        return Image.fromarray(self.cam_center
                               .stream_colour(320 * 160 * 4).reshape(160, 320, 4)).convert('RGB')


# TODO refactor all road classes
RoadNodes = list[tuple[float, float, float, float]]
List2DTuple = list[tuple[float, float]]


class BeamNGRoad:
    def __init__(self, road_nodes: RoadNodes):
        self.road_nodes: RoadNodes = road_nodes
        self.decal_road: DecalRoad = DecalRoad('street_1').add_4d_points(road_nodes)
        self.road_points: RoadPoints = RoadPoints().add_middle_nodes(road_nodes)
        self.waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(road_nodes[-1]))
        self.install()

    def vehicle_start_pose(self):
        return self.road_points.vehicle_start_pose()

    def decal_to_json(self):
        return self.decal_road.to_json()

    def install(self):
        maps.beamng_map.generated().write_items(self.decal_to_json() + '\n' + self.waypoint_goal.to_json())


class RoadPoints:

    @classmethod
    def from_nodes(cls, middle_nodes: RoadNodes):
        res = RoadPoints()
        res.add_middle_nodes(middle_nodes)
        return res

    def __init__(self):
        self.middle: RoadNodes = []
        self.right: List2DTuple = []
        self.left: List2DTuple = []
        self.n = 0

    def add_middle_nodes(self, middle_nodes):
        n = len(self.middle) + len(middle_nodes)

        assert n >= 2, f'At least, two nodes are needed'

        assert all(len(point) >= 4 for point in middle_nodes), \
            f'A node is a tuple of 4 elements (x,y,z,road_width)'

        self.n = n
        self.middle += list(middle_nodes)
        self.left += [None] * len(middle_nodes)
        self.right += [None] * len(middle_nodes)
        self._recalculate_nodes()
        return self

    def _recalculate_nodes(self):
        for i in range(self.n - 1):
            l, r = self.calc_point_edges(self.middle[i], self.middle[i + 1])
            self.left[i] = l
            self.right[i] = r

        # the last middle point
        self.right[-1], self.left[-1] = self.calc_point_edges(self.middle[-1], self.middle[-2])

    @classmethod
    def calc_point_edges(cls, p1, p2) -> tuple[tuple, tuple]:
        origin = np.array(p1[0:2])

        a = np.subtract(p2[0:2], origin)

        # TODO: changed from 2 to 4
        # calculate the vector which length is half the road width
        v = (a / np.linalg.norm(a)) * p1[3] / 2
        # add normal vectors
        l = origin + np.array([-v[1], v[0]])
        r = origin + np.array([v[1], -v[0]])
        return tuple(l), tuple(r)

    def vehicle_start_pose(self, meters_from_road_start=2.5, road_point_index=0) \
            -> BeamNGPose:
        assert self.n > road_point_index, f'road length is {self.n} it does not have index {road_point_index}'
        p1 = self.middle[road_point_index]
        p1r = self.right[road_point_index]
        p2 = self.middle[road_point_index + 1]

        p2v = np.subtract(p2[0:2], p1[0:2])
        v = (p2v / np.linalg.norm(p2v)) * meters_from_road_start
        # TODO: test the amount
        origin = np.add(p1[0:2], p1r[0:2]) / 2
        deg = np.degrees(np.arctan2([-v[0]], [-v[1]]))
        res = BeamNGPose(pos=tuple(origin + v) + (p1[2],), rot=(0, 0, deg[0]))
        return res

    def new_imagery(self):
        from .beamng_road_imagery import BeamNGRoadImagery
        return BeamNGRoadImagery(self)

    def plot_on_ax(self, ax):
        def _plot_xy(points, color, linewidth):
            tup = list(zip(*points))
            ax.plot(tup[0], tup[1], color=color, linewidth=linewidth)

        ax.set_facecolor('#7D9051')  # green
        _plot_xy(self.middle, '#FEA952', linewidth=1)  # arancio
        _plot_xy(self.left, 'white', linewidth=1)
        _plot_xy(self.right, 'white', linewidth=1)
        ax.axis('equal')
