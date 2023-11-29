import os
import time
import traceback

import numpy as np

from core.evaluator import Evaluator
from core.folders import FOLDERS, SeedStorage
from core.log import get_logger
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_interface import BeamNGInterface
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_roads import BeamNGRoad
from self_driving.simulation_data import SimulationData, SimulationDataRecord
from self_driving.simulation_data_collector import SimulationDataCollector
from training.udacity_utils import preprocess

log = get_logger(__file__)


class BeamNGLocalEvaluator(Evaluator):
    """Executes a local BeamNG instance and uses it to evaluate members."""

    def __init__(self, config: BeamNGConfig):
        self.config = config
        self.bng: BeamNGInterface | None = None

        self.model_file = str(FOLDERS.trained_models_colab.joinpath(config.MODEL_FILE))
        if not os.path.exists(self.model_file):
            raise Exception(f'File {self.model_file} does not exist!')

        self.model = None
        self.speed_limit = config.MAX_SPEED

    def evaluate(self, member: BeamNGMember) -> None:
        if not member.needs_evaluation():
            log.info(f'{member} is already evaluated. skipping')
            return

        counter = 20
        attempt = 0
        while True:
            attempt += 1
            if attempt == counter:
                raise Exception('Exhausted attempts')
            if attempt > 1:
                log.info(f'RETRYING TO run simulation, attempt {attempt}')
            else:
                log.info(f'{member} BeamNG evaluation start')
            if attempt > 2:
                time.sleep(5)
            sim = self._run_simulation(member.road)
            if sim.info.success:
                break

        member.distance_to_frontier = sim.min_oob_distance()
        log.info(f'{member} BeamNG evaluation completed')

    def _run_simulation(self, road: BeamNGRoad) -> SimulationData:
        """Runs the simulation on a given road. Initializes connection with BeamNG, creates a simulation
        of the road with a single vehicle and loops. In the loop captures images from the vehicle camera and feeds them
        to the ML model, then send the controls to the simulation. The loop will break if the end of the road is
        reached, if the vehicle exits from its lane or if a max number of iterations is reached."""
        if not self.bng:
            self.bng = BeamNGInterface(self.config)

        self.bng.setup_road(road)

        self.bng.setup_vehicle()

        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = self.config.SIM_NAME.replace('$(id)', simulation_id)
        sim_data_collector = SimulationDataCollector(self.bng, simulation_name=name)

        sim_data_collector.get_simulation_data().start()
        try:
            self.bng.beamng_bring_up()

            self.bng.setup_scenario_camera()

            # Make the simulation run for 1 step to fill the car camera buffer
            self.bng.beamng_step(1)

            if not self.model:
                from keras.src.saving.saving_api import load_model
                self.model = load_model(self.model_file)

            iterations_count = 1000
            idx = 0
            while True:
                idx += 1
                if idx >= iterations_count:
                    sim_data_collector.save()
                    raise Exception('Timeout simulation ', sim_data_collector.name)

                vehicle_state = sim_data_collector.collect_current_data(oob_bb=False)

                if vehicle_state.dist_from_goal < 6.0:
                    break

                if vehicle_state.is_oob:
                    break

                img = self.bng.vehicle.capture_image()
                # img.save(f"../img/{datetime.now().strftime('%d-%m_%H-%M-%S-%f')[:-3]}.jpg")

                steering_angle, throttle = self.predict(img, vehicle_state)
                self.bng.vehicle.control(throttle=throttle, steering=steering_angle, brake=0)

                self.bng.beamng_step()

            sim_data_collector.get_simulation_data().end(success=True)
        except Exception as ex:
            sim_data_collector.get_simulation_data().end(success=False)
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            if self.config.SAVE_SIM_DATA:
                sim_data_collector.save()
                try:
                    sim_data_collector.take_car_picture_if_needed()
                except Exception as ex:
                    log.warning('Cannot take OOB vehicle picture:')
                    traceback.print_exception(type(ex), ex, ex.__traceback__)

            self._end_simulation()

        return sim_data_collector.simulation_data

    def _end_simulation(self):
        """Terminates the simulation of a road, readying the simulator for the next one."""
        self.bng.beamng_stop_scenario()

    def predict(self, image, car_state: SimulationDataRecord):
        """Uses the loaded model to predict the next steering command for the vehicle from the image
        captured by the camera."""
        try:
            image = np.asarray(image)

            image = preprocess(image)
            image = np.array([image])

            steering_angle = float(self.model.predict(image, batch_size=1))

            speed = car_state.vel_kmh
            if speed > self.speed_limit:
                self.speed_limit = self.config.MIN_SPEED  # slow down
            else:
                self.speed_limit = self.config.MAX_SPEED
            throttle = 1.0 - steering_angle ** 2 - (speed / self.speed_limit) ** 2
            return steering_angle, throttle
        except Exception as ex:
            log.error('Cannot predict steering angle:')
            traceback.print_exception(type(ex), ex, ex.__traceback__)


if __name__ == '__main__':
    inst = BeamNGLocalEvaluator(BeamNGConfig())

    seed_storage = SeedStorage('population_HQ1')
    for i in range(12):
        mbr = BeamNGMember.from_dict(seed_storage.load_json_by_index(i))
        mbr.clear_evaluation()
        inst.evaluate(mbr)
        print(mbr)

    inst.bng.beamng_close()
