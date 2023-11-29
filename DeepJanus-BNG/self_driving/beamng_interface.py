import os
import subprocess
import traceback

from PIL import Image
from beamngpy import BeamNGpy, Scenario
from beamngpy.logging import BNGError, BNGDisconnectedError
from beamngpy.sensors import Camera

from core.folders import SeedStorage
from core.log import get_logger
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_map_utils import LEVEL_NAME, map_utils
from self_driving.beamng_vehicles import BeamNGVehicle
from self_driving.beamng_roads import BeamNGRoad
from self_driving.simulation_data import SimulationParams

log = get_logger(__file__)


class BeamNGInterface:
    """Manages the BeamNGpy instance and the connection to the simulator"""

    def __init__(self, config: BeamNGConfig, road: BeamNGRoad = None):
        """Initializes the interface to the BeamNG simulator. The simulator will not be opened until a call to
        beamng_bring_up() is made."""
        self.config = config

        self.road: BeamNGRoad | None = None
        if road:
            self.setup_road(road)

        # Remove the BeamNG version from the user path, as it will be added automatically by BeamNG
        user_path = os.path.dirname(config.BEAMNG_USER_DIR)
        self._bng = BeamNGpy(config.BEAMNG_HOST, config.BEAMNG_PORT, user=user_path)

        self.vehicle: BeamNGVehicle | None = None

        self.scenario: Scenario | None = None
        self.camera: Camera | None = None

        self.params = SimulationParams(iteration_steps=config.BEAMNG_STEPS, fps_limit=config.BEAMNG_FPS)

        self.simulation_count = 0

    def setup_road(self, road: BeamNGRoad):
        """Sets up the list of nodes that represent the road to simulate."""
        map_utils.install_map_if_needed(self.config.BEAMNG_USER_DIR)
        self.road = road
        map_utils.install_road(road)

    def setup_vehicle(self, all_cameras=False) -> BeamNGVehicle:
        """Sets up the vehicle to use for the simulation. Parameter 'all_cameras' can be set to True if
        additional cameras moved to the left and right of the vehicle are needed."""
        self.vehicle = BeamNGVehicle(all_cameras)

        if self.road:
            self.vehicle.start_pose = self.road.vehicle_start_pose()

        return self.vehicle

    def setup_scenario_camera(self):
        """Sets up the global camera that will capture images of the car if it goes out of bounds."""
        assert self._bng.connection
        assert self.scenario
        self.camera = Camera('scenario_camera', self._bng, is_static=True, pos=(0, 0, 0), dir=(0, 0, 0),
                             field_of_view_y=90, resolution=(1280, 1280), requested_update_time=-1)

    def capture_image(self, position: tuple[float, float, float], direction: tuple[float, float, float]) -> Image:
        """Captures an image from the scenario camera."""
        self.camera.set_position(position)
        self.camera.set_direction(direction)

        req_id = self.camera.send_ad_hoc_poll_request()
        while not self.camera.is_ad_hoc_poll_request_ready(req_id):
            self.beamng_step(1)

        return self.camera.collect_ad_hoc_poll_request(req_id)['colour'].convert('RGB')

    def beamng_bring_up(self):
        """Connects to the simulator, opening a new instance if necessary. Then, sets up the scenario (road, vehicle,
        and sensors) and loads it into the simulation. Note that the scenario will be in a paused state and subsequent
        calls to step() are needed to actually simulate."""
        if not self._bng.connection:
            self._bng.open()
            # Do not pipe BeamNG.tech stdout to Python stdout
            self._bng.process.stdout = subprocess.DEVNULL

        self.scenario = Scenario(LEVEL_NAME, f'{LEVEL_NAME}_scenario')
        if self.vehicle:
            self.vehicle.add_to_scenario(self.scenario)

        self.scenario.make(self._bng)

        self._bng.load_scenario(self.scenario)
        self._bng.set_deterministic(self.params.fps_limit)
        self._bng.pause()

        if self.vehicle:
            self.vehicle.setup_cameras(self._bng)

        self._bng.start_scenario()

    def beamng_step(self, steps: int = None):
        """Runs the simulation for the predefined number of steps. A custom number of steps can be passed as
         an argument."""
        if not steps:
            steps = self.params.iteration_steps
        self._bng.step(steps)

    def beamng_stop_scenario(self):
        """Stops and exits the simulation scenario without closing the simulator."""
        try:
            self._bng.stop_scenario()
        except BNGError:
            pass  # No scenario is loaded
        except Exception as ex:
            log.warning('Cannot stop BeamNG scenario:')
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        self.scenario = None

        self.simulation_count += 1
        # If set in the config, restart the simulator after n runs (useful to circumvent BeamNG memory leaks)
        if self.config.BEAMNG_RESTART_AFTER > 0:
            if self.simulation_count >= self.config.BEAMNG_RESTART_AFTER:
                self.beamng_close()
                self.simulation_count = 0

    def beamng_close(self):
        """Closes the simulator."""
        if self._bng:
            try:
                self._bng.close()
            except BNGDisconnectedError:
                pass  # We want to disconnect
            except Exception as ex:
                log.warning('Cannot close BeamNG instance:')
                traceback.print_exception(type(ex), ex, ex.__traceback__)


if __name__ == '__main__':
    brewer = BeamNGInterface(BeamNGConfig())

    seed_storage = SeedStorage('population_HQ1')
    member = BeamNGMember.from_dict(seed_storage.load_json_by_index(0))

    brewer.setup_road(member.road)
    brewer.setup_vehicle()

    brewer.beamng_bring_up()
    print('bring up ok')
    brewer.beamng_step(10)
    print('advanced 10 steps')
    input('waiting keypress...')
    print('key received')
    brewer.beamng_close()
