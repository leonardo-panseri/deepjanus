from beamngpy.sensors import Camera
from beamngpy import BeamNGpy, Vehicle


class BeamNGCarCameras:
    def __init__(self, beamng: BeamNGpy, vehicle: Vehicle):
        self.direction = (0, -1, 0)
        self.fov = 120
        self.resolution = (320, 160)
        self.y = -2.2
        self.z = 1.0
        self.beamng = beamng
        self.vehicle = vehicle

    def setup_cameras(self):
        self.cam_center = Camera('cam_center', self.beamng, self.vehicle, pos=(-0.3, self.y, self.z),
                                 dir=self.direction, field_of_view_y=self.fov, resolution=self.resolution,
                                 requested_update_time=0.1, is_using_shared_memory=True, is_render_annotations=False,
                                 is_render_depth=False, is_streaming=True)
        self.cam_left = Camera('cam_left', self.beamng, self.vehicle, pos=(-1.3, self.y, self.z),
                               dir=self.direction, field_of_view_y=self.fov, resolution=self.resolution,
                               requested_update_time=0.1)
        self.cam_right = Camera('cam_right', self.beamng, self.vehicle, pos=(0.4, self.y, self.z),
                                dir=self.direction, field_of_view_y=self.fov, resolution=self.resolution,
                                requested_update_time=0.1)
