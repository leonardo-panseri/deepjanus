from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from beamngpy.misc.quat import angle_to_quat
from core.folder_storage import SeedStorage
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_member import BeamNGMember

from self_driving.decal_road import DecalRoad
from self_driving.road_points import List4DTuple, RoadPoints
from self_driving.simulation_data import SimulationParams
from self_driving.beamng_pose import BeamNGPose
from udacity_integration.beamng_car_cameras import BeamNGCarCameras


class BeamNGBrewer:
    def __init__(self, road_nodes: List4DTuple = None):
        self.beamng = BeamNGpy('localhost', 12345)
        self.vehicle: Vehicle = None
        self.use_camera = False
        self.camera: Camera = None
        self.car_cameras: BeamNGCarCameras = None
        if road_nodes:
            self.setup_road_nodes(road_nodes)
        steps = 5
        self.params = SimulationParams(beamng_steps=steps, delay_msec=int(steps * 0.05 * 1000))
        self.vehicle_start_pose = BeamNGPose()

        self.beamng.open()
        self.beamng.logger.setLevel(20)

    def setup_road_nodes(self, road_nodes):
        self.road_nodes = road_nodes
        self.decal_road: DecalRoad = DecalRoad('street_1').add_4d_points(road_nodes)
        self.road_points = RoadPoints().add_middle_nodes(road_nodes)

    def setup_vehicle(self) -> Vehicle:
        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='TIG', color='Red')
        self.vehicle.attach_sensor('gforces', GForces())
        self.vehicle.attach_sensor('electrics', Electrics())
        self.vehicle.attach_sensor('damage', Damage())
        self.vehicle.attach_sensor('timer', Timer())
        return self.vehicle

    def setup_car_cameras(self) -> BeamNGCarCameras:
        self.car_cameras = BeamNGCarCameras(self.beamng, self.vehicle)
        return self.car_cameras

    def setup_scenario_camera(self, resolution=(1280, 1280), fov=120) -> Camera:
        assert self.use_camera is False
        self.use_camera = True
        return self.camera

    def bring_up(self):
        self.scenario = Scenario('tig', 'tigscenario')
        if self.vehicle:
            self.scenario.add_vehicle(self.vehicle, pos=self.vehicle_start_pose.pos,
                                 rot_quat=angle_to_quat(self.vehicle_start_pose.rot))

        self.scenario.make(self.beamng)

        self.beamng.load_scenario(self.scenario)
        self.beamng.set_deterministic(60)
        self.beamng.pause()

        if self.use_camera:
            self.camera = Camera('brewer_camera', self.beamng, is_static=True, pos=(0, 0, 0), dir=(0, 0, 0), field_of_view_y=120,
                                 resolution=(1280, 1280))

        if self.car_cameras:
            self.car_cameras.setup_cameras()

        self.beamng.start_scenario()

    def __del__(self):
        if self.beamng:
            try:
                self.beamng.close()
            except:
                pass


if __name__ == '__main__':
    config = BeamNGConfig()
    brewer = BeamNGBrewer()
    vehicle = brewer.setup_vehicle()
    camera = brewer.setup_scenario_camera()

    seed_storage = SeedStorage('basic5')
    member = BeamNGMember.from_dict(seed_storage.load_json_by_index(1))
    brewer.setup_road_nodes(member.sample_nodes)
    brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

    brewer.bring_up()
    print('bring up ok')
    brewer.beamng.resume()
    print('resumed')
    input('waiting keypress...')
    print('key received')
    brewer.beamng.stop_scenario()
