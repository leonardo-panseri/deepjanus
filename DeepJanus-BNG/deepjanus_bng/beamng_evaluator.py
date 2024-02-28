import json
import logging
import os
import time
import traceback
from typing import TYPE_CHECKING

import numpy as np

from deepjanus.evaluator import Evaluator, ParallelEvaluator
from .beamng_config import BeamNGConfig
from .beamng_interface import BeamNGInterface
from .beamng_roads import BeamNGRoad
from .simulation_data import SimulationData, SimulationDataRecord
from .simulation_data_collector import SimulationDataCollector
from .training.training_utils import preprocess

if TYPE_CHECKING:
    from .beamng_member import BeamNGMember


config: BeamNGConfig
log: logging.Logger
bng: BeamNGInterface | None = None
model_file: str
model = None
speed_limit: float


class BeamNGLocalEvaluator(Evaluator):
    """Executes a local BeamNG instance and uses it to evaluate members."""

    def __init__(self, cfg: BeamNGConfig):
        global config
        config = cfg

        _initialize_globals(cfg)

    @staticmethod
    def evaluate_member_sequential(member: 'BeamNGMember') -> 'BeamNGMember':
        return _evaluate_member(member)


class BeamNGParallelEvaluator(ParallelEvaluator):
    """Executes parallel BeamNG instances in a process pool and uses them to evaluate members."""

    def __init__(self, cfg: BeamNGConfig):
        self.project_root = str(cfg.FOLDERS.root)
        _initialize_globals(cfg)

        self.initial_port = cfg.BEAMNG_PORT
        self.initial_path = cfg.BEAMNG_USER_DIR

        self.ports = []
        self.paths = []

        super().__init__(cfg.PARALLEL_EVALS)

    def setup_worker_init_args(self, args_queue):
        for i in range(self.num_workers):
            # Generate port where the instance of the simulator will run
            port = self.initial_port + i
            self.ports.append(port)

            # User content folder that the instance of the simulator will use
            # Instances need to have different user folders to avoid conflicts in accessing files
            path = self.initial_path.replace('instance0', f'instance{i}')

            if not os.path.exists(path):
                # Set the simulation to run without online features to prevent problems
                cloud_settings_path = os.path.join(path, 'settings', 'cloud')
                os.makedirs(cloud_settings_path, exist_ok=True)
                with open(os.path.join(cloud_settings_path, 'settings.json'), 'w') as f:
                    f.write(json.dumps({
                        "onlineFeatures": "disable",
                        "telemetry": "disable"
                    }))

            self.paths.append(path)
            args_queue.put((i, port, path, self.project_root))

    @staticmethod
    def init_worker(args_queue):
        global config, log

        parallel_index, port, path, project_root = args_queue.get()

        # Disable TensorFlow logs
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        import tensorflow.python.util.module_wrapper as mw
        mw._PER_MODULE_WARNING_LIMIT = 0

        log = logging.getLogger('beamngpy')
        log.setLevel(logging.DEBUG)
        
        # Create a config with the settings for the instance of the simulator
        config = BeamNGConfig(project_root)
        config.BEAMNG_PORT = port
        config.BEAMNG_USER_DIR = path

        _initialize_globals(config)

    @staticmethod
    def evaluate_member_parallel(member: 'BeamNGMember', stop_workers_event) -> 'BeamNGMember':
        return _evaluate_member(member, stop_workers_event)


def _initialize_globals(cfg: BeamNGConfig):
    global model_file, speed_limit

    model_file = str(cfg.FOLDERS.models.joinpath(cfg.MODEL_FILE))
    if not os.path.exists(model_file):
        raise Exception(f'File {model_file} does not exist!')
    speed_limit = cfg.MAX_SPEED


def _evaluate_member(member: 'BeamNGMember', stop_workers_event=None) -> 'BeamNGMember':
    if log.handlers:
        h_old = log.handlers[0]
        h_old.close()
        log.removeHandler(h_old)
    log_file = os.path.join(os.path.dirname(config.BEAMNG_USER_DIR), 'sim.log')
    h = logging.FileHandler(log_file, 'w')
    h.setFormatter(logging.Formatter(rf'[%(asctime)s %(levelname)s] %(message)s', '%H:%M:%S'))
    log.addHandler(h)

    log.info(f'Evaluating {member}')

    sim = _run_simulation(member.road, stop_workers_event)

    # Requirement: do not go outside lane boundaries
    satisfy_requirements = sim is not None and sim.min_oob_distance() > 0

    # Update member here to ensure that log contains evaluation info
    member.satisfy_requirements = satisfy_requirements

    return member


def _run_simulation(road: BeamNGRoad, stop_workers_event) -> SimulationData | None:
    """Runs the simulation on a given road. Initializes connection with BeamNG, creates a simulation
    of the road with a single vehicle and loops. In the loop captures images from the vehicle camera and feeds them
    to the ML model, then send the controls to the simulation. The loop will break if the end of the road is
    reached, if the vehicle exits from its lane or if a max number of iterations is reached."""
    global bng, model

    if stop_workers_event is not None and stop_workers_event.is_set():
        _end_simulation()
        return None

    if not bng:
        bng = BeamNGInterface(config)

    bng.setup_road(road)

    bng.setup_vehicle()

    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
    name = config.SIM_NAME.replace('$(id)', simulation_id)
    sim_data_collector = SimulationDataCollector(bng, simulation_name=name)

    sim_data_collector.get_simulation_data().start()

    bng.beamng_bring_up()

    # bng.setup_scenario_camera()

    # Make the simulation run for 1 step to fill the car camera buffer
    bng.beamng_step(1)

    if not model:
        import tensorflow as tf
        model = tf.saved_model.load(model_file)

    iterations_count = 1000
    idx = 0
    while True:
        if stop_workers_event is not None and stop_workers_event.is_set():
            _end_simulation()
            return None

        idx += 1
        if idx >= iterations_count:
            sim_data_collector.save()
            raise Exception('Timeout simulation ', sim_data_collector.name)

        vehicle_state = sim_data_collector.collect_current_data(oob_bb=False)

        if vehicle_state.dist_from_goal < 6.0:
            break

        if vehicle_state.is_oob:
            break

        img = bng.vehicle.capture_image()
        # img.save(f"../img/{datetime.now().strftime('%d-%m_%H-%M-%S-%f')[:-3]}.jpg")

        steering_angle, throttle = _predict(img, vehicle_state)
        bng.vehicle.control(throttle=throttle, steering=steering_angle, brake=0)

        bng.beamng_step()

    sim_data_collector.get_simulation_data().end(success=True)

    if config.SAVE_SIM_DATA:
        sim_data_collector.save()
        sim_data_collector.take_car_picture_if_needed()

    _end_simulation()

    return sim_data_collector.simulation_data


def _end_simulation():
    """Terminates the simulation of a road, readying the simulator for the next one."""
    bng.beamng_stop_scenario()


def _predict(image, car_state: SimulationDataRecord):
    """Uses the loaded model to predict the next steering command for the vehicle from the image
    captured by the camera."""
    global speed_limit

    image = np.asarray(image)

    image = preprocess(image).astype(dtype=np.float32)
    image = np.array([image])

    steering_angle = float(model(image).numpy())

    speed = car_state.vel_kmh
    if speed > speed_limit:
        speed_limit = config.MIN_SPEED  # slow down
    else:
        speed_limit = config.MAX_SPEED
    throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2
    return steering_angle, throttle
