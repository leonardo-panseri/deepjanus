import json
import math
import multiprocessing
import threading
import timeit
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

from core.folders import FOLDERS
from core.individual import Individual
from core.log import get_logger
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_interface import BeamNGInterface
from self_driving.beamng_member import BeamNGMember

if TYPE_CHECKING:
    from self_driving.beamng_problem import BeamNGProblem

log = get_logger(__file__)


class BeamNGIndividual(Individual[BeamNGMember]):
    """Individual for DeepJanus-BNG"""

    def __init__(self, mbr: BeamNGMember, seed: BeamNGMember = None,
                 neighbors: list[BeamNGMember] = None, name: str = None):
        """Creates a DeepJanus-BNG individual. Parameter 'name' can be passed to create clones of existing individuals,
        disabling the automatic incremental names."""
        super().__init__(mbr, seed, neighbors, name)

    def clone(self, individual_creator) -> 'BeamNGIndividual':
        # Need to use the DEAP creator to instantiate new individual
        # Do not pass self.name, as we use this to create the offspring
        res: BeamNGIndividual = individual_creator(self.mbr.clone(), self.seed)
        log.info(f'Cloned to {res} from {self}')
        return res

    def save(self, folder: Path):
        # Save a JSON representation of the individual
        json_path = folder.joinpath(self.name + '.json')
        json_path.write_text(json.dumps(self.to_dict()))

        nbh_size = len(self.neighbors)

        # Save an image of member road and all neighbors
        num_cols = 3
        num_rows = math.ceil(nbh_size / num_cols) + 1
        fig = plt.figure()
        gs = fig.add_gridspec(num_rows, num_cols)
        fig.set_size_inches(15, 10)

        def plot(member: BeamNGMember, pos: plt.SubplotSpec):
            ax = fig.add_subplot(pos)
            ax.set_title(f'{member}', fontsize=12)
            member.to_image(ax)

        plot(self.mbr, gs[0, 1])
        for i in range(nbh_size):
            row = math.floor(i / num_cols) + 1
            col = i % num_cols
            plot(self.neighbors[i], gs[row, col])

        fig.suptitle(f'Neighborhood size = {nbh_size}; Frontier distance = {self.distance_to_frontier}')
        fig.savefig(folder.joinpath(self.name + '_neighborhood.svg'))
        plt.close(fig)

    def to_dict(self) -> dict:
        return {'name': self.name,
                'frontier_dist': self.distance_to_frontier,
                'mbr': self.mbr.to_dict(),
                'neighbors': [nbh.to_dict() for nbh in self.neighbors],
                'seed': self.seed.to_dict() if self.seed else None}

    @classmethod
    def from_dict(cls, d: dict, individual_creator) -> 'BeamNGIndividual':
        mbr = BeamNGMember.from_dict(d['mbr'])
        neighbors = [BeamNGMember.from_dict(nbh) for nbh in d['neighbors']]
        seed = BeamNGMember.from_dict(d['seed']) if d['seed'] else None
        ind: BeamNGIndividual = individual_creator(mbr, seed, neighbors, d['name'])
        ind.distance_to_frontier = d['frontier_dist']
        return ind

    def evaluate(self, problem: 'BeamNGProblem') -> tuple[float, float]:
        num_parallel = problem.config.PARALLEL_EVALS

        # Check if we need to parallelize
        if num_parallel < 2:
            return super().evaluate(problem)

        log.info(f'Starting evaluation of {self}')
        start = timeit.default_timer()

        self.sparseness = problem.archive.evaluate_sparseness(self)

        unsafe_count = 0
        # Evaluate original member
        if not self.mbr.evaluate(problem.get_evaluator()):
            unsafe_count += 1

        curr_neighbors = 0
        max_neighbors = problem.config.MAX_NEIGHBORS
        curr_err = 0
        desired_error = problem.config.TARGET_ERROR

        # Empty map
        self.neighbors_hash = {}

        confidence_level = problem.config.CONFIDENCE_LEVEL
        lower_bound: float | None = None
        upper_bound: float | None = None

        cond_exploration_complete = threading.Condition()
        # Put arguments for each worker in a multiprocess queue
        # Each worker will initialize itself with a different set of arguments from the queue
        args_queue = multiprocessing.Queue()
        ports = []
        userpaths = []
        for i in range(problem.config.PARALLEL_EVALS):
            # Generate port where the instance of the simulator will run
            port = problem.config.BEAMNG_PORT + i
            ports.append(port)
            # Generate user content folder that the instance of the simulator will use
            # Instances need to have different user folders to avoid conflicts in accessing files
            userpath = FOLDERS.simulations.joinpath('beamng_parallel', f'{i}', '0.30')
            userpath = str(userpath)
            userpaths.append(userpath)

            args_queue.put((i, port, userpath))
        pool = multiprocessing.Pool(problem.config.PARALLEL_EVALS, init_worker, (args_queue, eval_neighbor))

        def neighbor_eval_completed(result: tuple[BeamNGMember, int]):
            # Make sure that only one result can be processed at a time
            with cond_exploration_complete:
                neighbor, index = result
                # Log here, as logs from other processes won't be shown
                log.info(f'{neighbor} BeamNG evaluation completed')
                
                nonlocal unsafe_count, curr_neighbors, lower_bound, upper_bound, curr_err
                if not neighbor.satisfy_requirements:
                    unsafe_count += 1

                curr_neighbors += 1
                self.neighbors.append(neighbor)

                # Number of evaluated members, all generated neighbors plus the original member
                evaluated = curr_neighbors + 1
                # Calculate Wilson Confidence Interval based on the estimator
                lower_bound, upper_bound = self._calculate_wilson_ci(unsafe_count / evaluated, evaluated,
                                                                     confidence_level)
                curr_err = (upper_bound - lower_bound) / 2.
                log.info(f'Evaluated {curr_neighbors}. '
                         f'CI is now [{lower_bound:.3f},{upper_bound:.3f}] (err: +-{curr_err:.3f})')

                # If desired precision is reached or the maximum number of neighbors is reached, stop exploration
                if curr_err <= desired_error or curr_neighbors == max_neighbors:
                    pool.terminate()
                    cond_exploration_complete.notify()
                    return

                # If the maximum number of neighbors have not been reached generate a new one and evaluate it
                if curr_neighbors < max_neighbors:
                    new_index = index + problem.config.PARALLEL_EVALS
                    new_neighbor = self.generate_neighbor(problem, new_index)
                    log.info(f'{new_neighbor} BeamNG evaluation start')
                    pool.apply_async(eval_neighbor, (new_neighbor, new_index),
                                     callback=neighbor_eval_completed,
                                     error_callback=lambda e: log.error(f'[NBR{new_index+1}] %s', e))

        # Start evaluation of first n neighbors
        for i in range(problem.config.PARALLEL_EVALS):
            # Generate the initial neighbor that the instance of the simulator will evaluate
            # and submit it to one process of the pool
            nbr = self.generate_neighbor(problem, i)
            log.info(f'{nbr} BeamNG evaluation start')
            pool.apply_async(eval_neighbor, (nbr, i),
                             callback=neighbor_eval_completed,
                             error_callback=lambda e: log.error(f'[NBR{i+1}] %s', e))

        # Wait for neighborhood exploration to end
        with cond_exploration_complete:
            cond_exploration_complete.wait()

        # Close all the instances of BeamNG used for neighborhood evaluation, but the main one
        for i in range(1, problem.config.PARALLEL_EVALS):
            cfg = BeamNGConfig()
            cfg.BEAMNG_PORT = ports[i]
            bng = BeamNGInterface(cfg)
            bng.beamng_open(launch=False)
            bng.beamng_close()

        pool.join()

        self.distance_to_frontier = (lower_bound, upper_bound)

        # Fitness function 'Quality of Individual'
        ff1 = self.sparseness
        # Fitness function 'Distance to Frontier'
        p_th = problem.config.PROBABILITY_THRESHOLD
        ff2 = max(abs(upper_bound - p_th), abs(lower_bound - p_th)) / max(p_th, 1 - p_th)

        minutes, seconds = divmod(timeit.default_timer() - start, 60)
        log.info(f'Time for eval: {int(minutes):02}:{int(seconds):02}')
        log.info(f'Evaluated {self}')
        return ff1, ff2


def init_worker(args_queue: multiprocessing.Queue, eval_fun):
    parallel_index, port, userpath = args_queue.get()

    import logging
    from logging import FileHandler
    from pathlib import Path
    log_file = Path(userpath).parent.joinpath('sim.log')
    h = FileHandler(log_file, 'w')
    h.setFormatter(logging.Formatter(rf'[%(asctime)s %(levelname)s] %(message)s', '%H:%M:%S'))
    bng_log = logging.getLogger('beamngpy')
    bng_log.setLevel(logging.INFO)
    bng_log.addHandler(h)

    bng_log.info(f'Starting parallel BeamNG instance {parallel_index} on port {port}')
    
    from self_driving.beamng_evaluator import BeamNGLocalEvaluator
    from self_driving.beamng_config import BeamNGConfig
    # Set the simulation to run without online features to prevent problems
    cloud_settings = Path(userpath).joinpath('settings', 'cloud')
    cloud_settings.mkdir(parents=True, exist_ok=True)
    cloud_settings.joinpath('settings.json').write_text(json.dumps({
        "onlineFeatures": "disable",
        "telemetry": "disable"
    }))

    # Create a config with the settings for the instance of the simulator
    cfg = BeamNGConfig()
    cfg.BEAMNG_PORT = port
    cfg.BEAMNG_USER_DIR = userpath

    # Create the evaluator and launch the instance of the simulator
    evaluator = BeamNGLocalEvaluator(cfg)
    evaluator.bng = BeamNGInterface(cfg)
    evaluator.bng.beamng_open()

    # Save a reference to the evaluator in the function executed by the worker
    eval_fun.evaluator = evaluator


def eval_neighbor(neighbor: BeamNGMember, index: int):
    """Function to execute in a subprocess to evaluate a neighbor."""
    # Retrieve the evaluator instance created by the worker initialization function
    evaluator = eval_neighbor.evaluator
    neighbor.satisfy_requirements = evaluator.evaluate(neighbor)
    import logging
    logging.getLogger('beamngpy').info(f"================== Evaluation {index} done ==================")
    return neighbor, index
