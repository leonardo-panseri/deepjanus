import os
import time
import traceback
from typing import List, Tuple, Optional

from PIL import Image

from core.folder_storage import SeedStorage
from core.folders import folders
from core.log import get_logger
from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_evaluator import BeamNGEvaluator
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_tig_maps import maps
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.nvidia_prediction import NvidiaPrediction
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader

log = get_logger(__file__)

FloatDTuple = Tuple[float, float, float, float]


class BeamNGNvidiaOob(BeamNGEvaluator):
    def __init__(self, config: BeamNGConfig):
        self.config = config
        self.brewer: Optional[BeamNGBrewer] = None
        self.model_file = str(folders.trained_models_colab.joinpath(config.keras_model_file))
        if not os.path.exists(self.model_file):
            raise Exception(f'File {self.model_file} does not exist!')
        self.model = None
        self.simulation_count = 0

    def evaluate(self, members: List[BeamNGMember]):
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
                    log.info(f'RETRYING TO run simulation {attempt}')
                    # self._close()
                else:
                    log.info(f'{member} BeamNG evaluation start')
                if attempt > 2:
                    time.sleep(5)
                sim = self._run_simulation(member.sample_nodes)
                if sim.info.success:
                    break

            member.distance_to_boundary = sim.min_oob_distance()
            log.info(f'{member} BeamNG evaluation completed')

    def _run_simulation(self, nodes) -> SimulationData:
        if not self.brewer:
            self.brewer = BeamNGBrewer()
            self.camera = self.brewer.setup_scenario_camera()

        if self.simulation_count >= self.config.beamng_restart_after_n_simulations:
            self.brewer.close_beamng()
            self.simulation_count = 0
        self.simulation_count += 1

        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))
        # TODO Fix this (path is relative to where this function is called)
        # maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

        self.vehicle = self.brewer.setup_vehicle()
        brewer.setup_car_cameras()
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        self.vehicle_state_reader = VehicleStateReader(self.vehicle)

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = self.config.simulation_name.replace('$(id)', simulation_id)
        sim_data_collector = SimulationDataCollector(self.vehicle, beamng, brewer.decal_road, brewer.params,
                                                     vehicle_state_reader=self.vehicle_state_reader,
                                                     camera=self.camera,
                                                     simulation_name=name)

        sim_data_collector.get_simulation_data().start()
        try:
            brewer.bring_up()
            # Make the simulation run for 1 step to fill the car camera buffer
            beamng.step(1)

            from keras.models import load_model
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

                sim_data_collector.collect_current_data(oob_bb=False)
                last_state: SimulationDataRecord = sim_data_collector.states[-1]

                dist = points_distance(last_state.pos, waypoint_goal.position)
                if dist < 6.0:
                    break

                if last_state.is_oob:
                    break

                img = Image.fromarray(brewer.car_cameras.cam_center
                                      .stream_colour(320 * 160 * 4).reshape(160, 320, 4)).convert('RGB')
                # img.save(f"../img/{datetime.now().strftime('%d-%m_%H-%M-%S-%f')[:-3]}.jpg")

                steering_angle, throttle = predict.predict(img, last_state)
                self.vehicle.control(throttle=throttle, steering=steering_angle, brake=0)
                beamng.step(steps)

            sim_data_collector.get_simulation_data().end(success=True)
        except Exception as ex:
            sim_data_collector.get_simulation_data().end(success=False, exception=ex)
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            if self.config.simulation_save:
                sim_data_collector.save()
                try:
                    sim_data_collector.take_car_picture_if_needed()
                except:
                    pass

            self.end_iteration()

        return sim_data_collector.simulation_data

    def end_iteration(self):
        try:
            if self.config.beamng_close_at_iteration:
                self._close()
            else:
                if self.brewer:
                    self.brewer.beamng.stop_scenario()
        except Exception as ex:
            log.debug('end_iteration() failed:')
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    def _close(self):
        if self.brewer:
            try:
                self.brewer.beamng.close()
            except Exception as ex:
                log.debug('beamng.close() failed:')
                traceback.print_exception(type(ex), ex, ex.__traceback__)
            self.brewer = None


if __name__ == '__main__':
    cfg = BeamNGConfig()
    inst = BeamNGNvidiaOob(cfg)

    seed_storage = SeedStorage('basic5')
    for i in range(1, 11):
        mbr = BeamNGMember.from_dict(seed_storage.load_json_by_index(i))
        mbr.clear_evaluation()
        inst.evaluate([mbr])
        log.info(mbr)

