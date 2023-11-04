from collections import namedtuple
from typing import Optional

import numpy as np
from beamngpy import Vehicle
from beamngpy.vehicle import Sensors

VehicleStateProperties = ['timer', 'damage', 'pos', 'dir', 'vel', 'gforces', 'gforces2', 'steering', 'steering_input',
                          'brake', 'brake_input', 'throttle', 'throttle_input', 'throttleFactor', 'engineThrottle',
                          'wheelspeed', 'vel_kmh']

VehicleState = namedtuple('VehicleState', VehicleStateProperties)


class VehicleStateReader:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.state: Optional[VehicleState] = None
        self.sensors: Optional[Sensors] = None
        self.vehicle_state = {}

    def get_state(self) -> VehicleState:
        return self.state

    def get_vehicle_bbox(self) -> dict:
        return self.vehicle.get_bbox()

    def update_state(self):
        self.vehicle.poll_sensors()
        sensors = self.vehicle.sensors
        self.sensors = sensors
        st = self.vehicle.state

        ele = sensors['electrics']
        gforces = sensors['gforces']

        vel = tuple(st['vel'])
        self.state = VehicleState(timer=sensors['timer']['time']
                                  , damage=sensors['damage']['damage']
                                  , pos=tuple(st['pos'])
                                  , dir=tuple(st['dir'])
                                  , vel=vel
                                  , gforces=(gforces['gx'], gforces['gy'], gforces['gz'])
                                  , gforces2=(gforces['gx2'], gforces['gy2'], gforces['gz2'])
                                  , steering=ele.get('steering', None)
                                  , steering_input=ele.get('steering_input', None)
                                  , brake=ele.get('brake', None)
                                  , brake_input=ele.get('brake_input', None)
                                  , throttle=ele.get('throttle', None)
                                  , throttle_input=ele.get('throttle_input', None)
                                  , throttleFactor=ele.get('throttleFactor', None)
                                  , engineThrottle=ele.get('engineThrottle', None)
                                  , wheelspeed=ele.get('wheelspeed', None)
                                  , vel_kmh=int(round(np.linalg.norm(vel) * 3.6)))
