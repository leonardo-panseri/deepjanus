import math
import timeit
from statistics import NormalDist
# Workaround for keeping type hinting while avoiding circular imports
from typing import TYPE_CHECKING
from typing import TypeVar, Generic

from core.log import get_logger
from core.member import Member

if TYPE_CHECKING:
    from core.problem import Problem

log = get_logger(__file__)
T = TypeVar('T', bound=Member)


class Individual(Generic[T]):
    """Class representing an individual of the population"""

    counter = 1

    def __init__(self, mbr: T, seed: T = None, neighbors: list[T] = None, name: str = None):
        """Creates a DeepJanus individual. Parameter 'name' can be passed to create clones of existing individuals,
        disabling the automatic incremental names."""
        self.name: str = name if name else f'ind{str(Individual.counter)}'
        if not name:
            Individual.counter += 1

        self.mbr: T = mbr
        self.neighbors: list[T] = neighbors if neighbors else []
        self.seed: T | None = seed

        self.sparseness: float | None = None
        self.distance_to_frontier: tuple[float, float] | None = None

    def clone(self, individual_creator) -> 'Individual':
        """Creates a deep copy of the individual using the provided DEAP creator."""
        raise NotImplemented()

    def evaluate(self, problem: 'Problem') -> tuple[float, float]:
        """Evaluates the individual and returns the two fitness values."""
        log.info(f'Starting evaluation of {self}')
        start = timeit.default_timer()

        self.sparseness = problem.archive.evaluate_sparseness(self)
        stop = timeit.default_timer()
        log.info(f'Time to sparseness: {stop - start}; archive len: {len(problem.archive)}')

        unsafe_count = 0
        if not self.mbr.evaluate(problem.get_evaluator()):
            unsafe_count += 1

        curr_neighbors = 0
        max_neighbors = problem.config.MAX_NEIGHBORS
        curr_err = 1
        desired_error = problem.config.TARGET_ERROR

        # Map to quickly check if neighbors are all different from each other
        neighbors_hash = {}

        confidence_level = problem.config.CONFIDENCE_LEVEL
        lower_bound: float | None = None
        upper_bound: float | None = None
        while curr_neighbors < max_neighbors and curr_err > desired_error:
            # Generate a neighbor different from all other neighbors
            equal_nbr = True
            nbr: Member | None = None
            while equal_nbr:
                nbr = self.mbr.clone(f'nbr{curr_neighbors + 1}_{self.name.replace("ind", "")}')
                nbr.mutate(problem.get_mutator())
                equal_nbr = nbr.member_hash() in neighbors_hash
            neighbors_hash[nbr.member_hash()] = True
            self.neighbors.append(nbr)

            if not nbr.evaluate(problem.get_evaluator()):
                unsafe_count += 1

            curr_neighbors += 1

            # Number of evaluated members, all generated neighbors plus the original member
            evaluated = curr_neighbors + 1
            # Calculate Wilson Confidence Interval based on the estimator
            lower_bound, upper_bound = self._calculate_wilson_ci(unsafe_count / evaluated, evaluated, confidence_level)
            curr_err = (upper_bound - lower_bound) / 2.
            log.info(f'Neighbor {curr_neighbors} evaluated. CI is now [{lower_bound:.3f},{upper_bound:.3f}] '
                     f'(err: +-{curr_err:.3f})')

        self.distance_to_frontier = (lower_bound, upper_bound)
        stop = timeit.default_timer()
        log.info(f'Time to eval: {stop - start}')

        # Fitness function 'Quality of Individual'
        ff1 = self.sparseness
        # Fitness function 'Distance to Frontier'
        p_th = problem.config.PROBABILITY_THRESHOLD
        ff2 = max(abs(upper_bound - p_th), abs(lower_bound - p_th)) / max(p_th, 1 - p_th)

        stop = timeit.default_timer()
        log.info(f'Total Time: {stop - start}')
        log.info(f'Evaluated {self}')
        return ff1, ff2

    @classmethod
    def _calculate_wilson_ci(cls, estimator: float, sample_size: int, confidence_level: float):
        """Calculates the Wilson Confidence Interval for an estimator, given the sample size and the desired confidence
        level. Returns a tuple containing the lower and upper bounds of the CI."""
        z = NormalDist().inv_cdf((1 + confidence_level) / 2.)
        gamma = (z * z) / sample_size
        p = 1 / (1 + gamma) * (estimator + gamma / 2)
        offset = z / (1 + gamma) * math.sqrt(estimator * (1 - estimator) / sample_size + gamma / (4 * sample_size))
        lower_bound = p - offset
        upper_bound = p + offset

        return lower_bound, upper_bound

    def mutate(self, problem: 'Problem'):
        """Mutates the individual."""
        self.mbr.mutate(problem.get_mutator())
        log.info(f'Mutated {self.mbr}')

    def distance(self, i2: 'Individual') -> float:
        """Calculates the distance with another individual."""
        return self.mbr.distance(i2.mbr)

    def save(self, folder):
        """Saves a human-interpretable representation of the individual on disk."""
        raise NotImplemented()

    def to_dict(self) -> dict:
        """Serializes the individual into a dictionary that can be easily stored on disk."""
        raise NotImplemented()

    @classmethod
    def from_dict(cls, d, individual_creator) -> 'Individual':
        """Builds an individual from its serialized representation using the provided DEAP creator."""
        raise NotImplemented()

    def __str__(self):
        frontier_eval = 'na'
        if self.distance_to_frontier:
            lb = f'{self.distance_to_frontier[0]:.3f}'
            ub = f'{self.distance_to_frontier[1]:.3f}'

            if self.distance_to_frontier[0] >= 0:
                lb = '+' + lb
            if self.distance_to_frontier[1] >= 0:
                ub = '+' + ub

            frontier_eval = f'[{lb},{ub}]'
        # frontier_eval = frontier_eval.ljust(9)
        return (f'{self.name.ljust(6)} mbr[{self.mbr}] nbh[n={len(self.neighbors)}; '
                f'sr={len(list(filter(lambda n: n.satisfy_requirements, self.neighbors)))}] seed[{self.seed}] '
                f'f{frontier_eval}')
