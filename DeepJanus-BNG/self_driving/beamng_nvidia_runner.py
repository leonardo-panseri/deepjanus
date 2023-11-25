import os
import time
import traceback

from keras.models import load_model

from core.folders import FOLDERS, SeedStorage
from core.log import get_logger
from self_driving.beamng_interface import BeamNGInterface
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_evaluator import BeamNGEvaluator
from self_driving.beamng_member import BeamNGMember
from self_driving.nvidia_prediction import NvidiaPrediction
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import points_distance
from self_driving.beamng_wrappers import RoadNodes

log = get_logger(__file__)


class BeamNGNvidiaOob(BeamNGEvaluator):
    def __init__(self, config: BeamNGConfig):
        self.config = config
        self.bng: BeamNGInterface | None = None

        self.model_file = str(FOLDERS.trained_models_colab.joinpath(config.MODEL_FILE))
        if not os.path.exists(self.model_file):
            raise Exception(f'File {self.model_file} does not exist!')

        self.model = None

    def evaluate(self, members: list[BeamNGMember]):
        for member in members:
            if not member.needs_evaluation():
                log.info(f'{member} is already evaluated. skipping')
                continue
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
                sim = self._run_simulation(member.sample_nodes)
                if sim.info.success:
                    break

            member.distance_to_boundary = sim.min_oob_distance()
            log.info(f'{member} BeamNG evaluation completed')

    def _run_simulation(self, nodes: RoadNodes) -> SimulationData:
        if not self.bng:
            self.bng = BeamNGInterface(self.config)

        # TODO Fix this (path is relative to where this function is called)
        # maps.install_map_if_needed()
        self.bng.setup_road(nodes)

        self.bng.setup_vehicle(True)

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
                self.model = load_model(self.model_file)
            predict = NvidiaPrediction(self.model, self.config)

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

                img = self.bng.vehicle.cameras.capture_image_center()
                # img.save(f"../img/{datetime.now().strftime('%d-%m_%H-%M-%S-%f')[:-3]}.jpg")

                steering_angle, throttle = predict.predict(img, vehicle_state)
                self.bng.vehicle.control(throttle=throttle, steering=steering_angle, brake=0)

                self.bng.beamng_step()

            sim_data_collector.get_simulation_data().end(success=True)
        except Exception as ex:
            sim_data_collector.get_simulation_data().end(success=False, exception=ex)
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            if self.config.SAVE_SIM_DATA:
                sim_data_collector.save()
                try:
                    sim_data_collector.take_car_picture_if_needed()
                except Exception as ex:
                    log.warning('Cannot take OOB vehicle picture:')
                    traceback.print_exception(type(ex), ex, ex.__traceback__)

            self.end_iteration()

        return sim_data_collector.simulation_data

    def end_iteration(self):
        self.bng.beamng_stop_scenario()


if __name__ == '__main__':
    inst = BeamNGNvidiaOob(BeamNGConfig())

    seed_storage = SeedStorage('population_HQ1')
    for i in range(12):
        mbr = BeamNGMember.from_dict(seed_storage.load_json_by_index(i))
        mbr.clear_evaluation()
        inst.evaluate([mbr])
        print(mbr)

    inst.bng.beamng_close()

