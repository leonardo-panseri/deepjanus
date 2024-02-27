import math
from multiprocessing import Event, Manager, Pool, Queue
import threading
import timeit
from statistics import NormalDist
from typing import TYPE_CHECKING

from .log import get_logger

if TYPE_CHECKING:
    from .individual import Individual
    from .member import Member
    from .problem import Problem

log = get_logger(__file__)


class Evaluator:
    """Base class for implementing sequential strategies to evaluate individuals"""

    _NORMAL_DIST = NormalDist()

    # Stopping condition #1: maximum number of neighbors that can be generated
    max_neighbors: int
    # Stopping condition #2: error that we want to reach with the confidence interval
    target_error: float
    # Confidence level to calculate the confidence interval
    confidence_level: float

    # Number of members already evaluated
    evaluated: int
    # Number of evaluated members that did not satisfy requirements
    unsafe: int
    # Current error of the confidence interval
    error: float

    @staticmethod
    def evaluate_member_sequential(member: 'Member') -> 'Member':
        """Evaluates if a member satisfies the requirements of the problem."""
        raise NotImplementedError()

    def _reset(self, problem: 'Problem'):
        """Resets all variables to start evaluation of a new individual."""
        self.max_neighbors = problem.config.MAX_NEIGHBORS
        self.target_error = problem.config.TARGET_ERROR
        self.confidence_level = problem.config.CONFIDENCE_LEVEL

        self.evaluated = 0
        self.unsafe = 0
        self.error = 1.

    def evaluate_individual(self, individual: 'Individual', problem: 'Problem'):
        """Evaluates an individual and returns the two fitness values."""
        self._reset(problem)

        log.info(f'Starting evaluation of {individual}')
        start = timeit.default_timer()

        # Calculates sparseness of this individual wrt other individuals in the archive
        sparseness = problem.archive.evaluate_sparseness(self)
        individual.sparseness = sparseness

        # Generates neighborhood to calculate unsafe region confidence interval
        lower_bound, upper_bound = self._do_evaluation(individual, problem)
        individual.unsafe_region_probability = (lower_bound, upper_bound)

        # Fitness function 'Quality of Individual'
        ff1 = sparseness
        # Fitness function 'Distance to Frontier'
        p_th = problem.config.PROBABILITY_THRESHOLD
        ff2 = max(abs(upper_bound - p_th), abs(lower_bound - p_th)) / max(p_th, 1 - p_th)

        # Update fitness here to print new values in logs
        individual.fitness.values = (ff1, ff2)

        minutes, seconds = divmod(timeit.default_timer() - start, 60)
        log.info(f'Time for eval: {int(minutes):02}:{int(seconds):02}')
        log.info(f'Evaluated {individual}')
        return ff1, ff2

    def _do_evaluation(self, individual: 'Individual', problem: 'Problem') -> tuple[float, float]:
        """Method that implements the strategy to evaluate the unsafe region confidence interval for an individual."""
        # Evaluate original member
        self._evaluate_member(individual.mbr)

        # Generate a neighbor different from all other neighbors and evaluate it until one of the stopping
        # conditions is reached
        lower_bound = .0
        upper_bound = .0
        while self._check_condition(individual):
            nbr: Member = individual.generate_neighbor(problem)

            # Evaluate the generated neighbor
            self._evaluate_member(nbr)
            individual.neighbors.append(nbr)

            # Calculate Wilson Confidence Interval for the unsafe region
            lower_bound, upper_bound = self._calculate_wilson_ci()
            log.info(f'CI is now [{lower_bound:.3f},{upper_bound:.3f}] (err: +-{self.error:.3f})')
        return lower_bound, upper_bound

    def _check_condition(self, individual: 'Individual'):
        """Checks if the one of the stopping conditions is reached."""
        return len(individual.neighbors) < self.max_neighbors and self.error > self.target_error

    def _calculate_wilson_ci(self):
        """Calculates the Wilson Confidence Interval for the estimator of the unsafe region probability for an
        individual. Returns a tuple containing the lower and upper bounds of the CI."""
        estimator = self.unsafe / self.evaluated
        sample_size = self.evaluated
        confidence_level = self.confidence_level

        z = self._NORMAL_DIST.inv_cdf((1 + confidence_level) / 2.)
        gamma = (z * z) / sample_size
        p = 1 / (1 + gamma) * (estimator + gamma / 2)
        offset = z / (1 + gamma) * math.sqrt(estimator * (1 - estimator) / sample_size + gamma / (4 * sample_size))
        lower_bound = p - offset
        upper_bound = p + offset

        # Save the current error that will be checked for one of the stopping conditions
        self.error = offset

        return lower_bound, upper_bound

    def _evaluate_member(self, member: 'Member'):
        """Method that implements the strategy to evaluate a single member. It should call the user-defined function
        evaluate_member(member)."""
        log.info(f'{member} evaluation start')
        if not self.evaluate_member_sequential(member).satisfy_requirements:
            self.unsafe += 1
        self.evaluated += 1
        log.info(f'{member} evaluation completed')


class ParallelEvaluator(Evaluator):
    """Base class for implementing parallel strategies to evaluate individuals"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

        # Queue that can be safely shared between processes
        # It is used to pass the initialization arguments to worker processes in the pool
        self.init_args_queue: Queue = Queue()
        # Context manager used to create an event that can be safely shared between processes
        self.manager: Manager = Manager()
        # Event that is shared with worker processes in the pool to notify when the workers should stop their work
        self.stop_workers_event: Event = self.manager.Event()

        # Add all necessary initialization arguments for worker processes to the queue
        self.setup_worker_init_args(self.init_args_queue)

        # Process pool that manages worker processes that will perform parallel evaluations of members
        self.process_pool: Pool = Pool(num_workers, self.init_worker, (self.init_args_queue,))

        # Error generated by worker process
        self.worker_error = None
        # Condition variable to signal when all the workers are done
        self.done_condition = threading.Condition()
        # Hash of the members that are currently being evaluated by worker processes
        self.current_evals = {}

        # Properties that need to be accessible by callbacks and the main thread
        self.individual: 'Individual'
        self.problem: 'Problem'
        self.lower_bound: float | None
        self.upper_bound: float | None

    def setup_worker_init_args(self, args_queue: Queue):
        """Adds all necessary initialization arguments for worker processes to the queue. Each worker process will
        then be able to retrieve their arguments from the initialization function."""
        pass

    @staticmethod
    def init_worker(args_queue: Queue):
        """Initialization function for worker processes in the pool."""
        pass

    @staticmethod
    def evaluate_member_parallel(member: 'Member', stop_workers_event: Event) -> 'Member':
        raise NotImplementedError()

    def _reset(self, problem: 'Problem'):
        super()._reset(problem)
        self.stop_workers_event.clear()
        self.lower_bound = None
        self.upper_bound = None

    def _stop_workers(self):
        """Sends an event to all worker processes to make them stop their current task."""
        self.stop_workers_event.set()

    def _close_pool(self):
        """Makes all the worker processes stop their current task and closes the pool when they are all done."""
        self._stop_workers()
        self.process_pool.close()
        self.process_pool.join()

    def _do_evaluation(self, individual: 'Individual', problem: 'Problem') -> tuple[float, float]:
        self.individual = individual
        self.problem = problem

        # Start evaluation of main member
        self._evaluate_member(individual.mbr)

        # Start evaluation of first neighbors
        for _ in range(problem.config.PARALLEL_EVALS - 1):
            nbr = individual.generate_neighbor(problem)
            self._evaluate_member(nbr)

        # Waits for all worker processes to be done
        with self.done_condition:
            self.done_condition.wait()

        # If a worker process raised an error, raise it from the main process
        if self.worker_error:
            raise self.worker_error

        return self.lower_bound, self.upper_bound

    def _on_eval_error(self, error):
        """Callback for errors raised by worker processes."""
        with self.done_condition:
            # If an error occurred in one of the other workers do nothing
            if self.worker_error:
                return

            # If a worker raised an error, save it and stop the evaluation
            self.worker_error = error
            self._close_pool()
            self.done_condition.notify()

    def _on_eval_complete(self, member: 'Member'):
        """Callback for tasks completed by worker processes."""
        with self.done_condition:
            # If an error occurred in one of the other workers do nothing
            if self.worker_error:
                return

            # Remove the hash of the member from the cache of members currently being evaluated
            self.current_evals.pop(member.member_hash())

            # If we already reached the stop condition do nothing, unless this is the last member
            # being evaluated that should wake up the main thread
            if self.stop_workers_event.is_set():
                if not self.current_evals:
                    self.done_condition.notify()
                return

            self.evaluated += 1
            if not member.satisfy_requirements:
                self.unsafe += 1
            log.info(f'{member} evaluation completed')

            # If this is not the main member of the individual, update the CI
            if member.name.startswith('nbr'):
                self.individual.neighbors.append(member)

                lower_bound, upper_bound = self._calculate_wilson_ci()
                log.info(f'CI is now [{lower_bound:.3f},{upper_bound:.3f}] (err: +-{self.error:.3f})')

                self.lower_bound = lower_bound
                self.upper_bound = upper_bound

            # If the stopping condition has not been reached, generate a new neighbor and start its evaluation
            if self._check_condition(self.individual):
                # Actually generate a neighbor only if the maximum number of neighbors has not already been reached
                # We have to take into account also the neighbors currently being evaluated by other workers
                if len(self.individual.neighbors_hash) < self.max_neighbors:
                    nbr = self.individual.generate_neighbor(self.problem)
                    self._evaluate_member(nbr)
            # Else stop the evaluation
            else:
                self._stop_workers()
            
            # If this is the last neighbor being evaluated wake up the main thread
            if not self.current_evals:
                self.done_condition.notify()

    def _evaluate_member(self, member: 'Member'):
        def on_eval_error(error):
            self._on_eval_error(error)

        def on_eval_complete(mbr):
            self._on_eval_complete(mbr)

        log.info(f'{member} evaluation start')
        self.current_evals[member.member_hash()] = True
        self.process_pool.apply_async(self.evaluate_member_parallel, (member, self.stop_workers_event),
                                      callback=on_eval_complete, error_callback=on_eval_error)
