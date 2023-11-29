from collections import namedtuple

import numpy as np
from PIL import Image
from beamngpy import angle_to_quat, Vehicle, Scenario, BeamNGpy
from beamngpy.sensors import GForces, Electrics, Damage, Timer, Camera
from beamngpy.vehicle import Sensors


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

    def capture_image_front(self) -> Image:
        """Captures an image from the front-facing camera mounted on the vehicle."""
        assert self.cameras, 'Vehicle cameras are disabled'
        return self.cameras.capture_image_center()

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

    def setup_cameras(self, bng: BeamNGpy, direction=(0, -1, 0), fov=120, resolution=(320, 160), y=-2.2, z=1.3):
        """Creates cameras in the simulation. Note that the simulation must be up and running for this to
        succeed."""
        self.cam_center = Camera('cam_center', bng, self.vehicle, pos=(-0.388, y, z),
                                 dir=direction, field_of_view_y=fov, resolution=resolution,
                                 requested_update_time=0.01, is_using_shared_memory=True, is_render_annotations=False,
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
