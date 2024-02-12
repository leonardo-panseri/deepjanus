import json
import math
import timeit
from statistics import NormalDist
from typing import TYPE_CHECKING
from typing import TypeVar, Generic

import matplotlib.pyplot as plt

from .log import get_logger
from .member import Member

if TYPE_CHECKING:
    from .problem import Problem

log = get_logger(__file__)
T = TypeVar('T', bound=Member)


class Individual(Generic[T]):
    """Class representing an individual of the population"""

    counter = 1
    _NORMAL_DIST = NormalDist()

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
        self.unsafe_region_probability: tuple[float, float] | None = None

        # Map to quickly check if neighbors are all different from each other
        self.neighbors_hash = {}

    def clone(self, individual_creator):
        """Creates a deep copy of the individual using the provided DEAP creator."""
        # Need to use the DEAP creator to instantiate new individual
        # Do not pass self.name, as we use this to create the offspring
        res = individual_creator(self.mbr.clone(), self.seed)
        log.info(f'Cloned to {res} from {self}')
        return res

    def generate_neighbor(self, problem: 'Problem', index: int) -> Member:
        """Generate a neighbor of this individual different from all other neighbors already generated."""
        equal_nbr = True
        nbr: Member | None = None
        while equal_nbr:
            nbr = self.mbr.clone(f'nbr{index + 1}_{self.name.replace("ind", "")}')
            nbr.mutate(problem.get_mutator())
            equal_nbr = nbr.member_hash() in self.neighbors_hash
        self.neighbors_hash[nbr.member_hash()] = True
        return nbr

    def evaluate(self, problem: 'Problem') -> tuple[float, float]:
        """Evaluates the individual and returns the two fitness values."""
        log.info(f'Starting evaluation of {self}')
        start = timeit.default_timer()

        self.sparseness = problem.archive.evaluate_sparseness(self)

        unsafe_count = 0
        # Evaluate original member
        if not self.mbr.evaluate(problem.get_evaluator()):
            unsafe_count += 1

        curr_neighbors = 0
        max_neighbors = problem.config.MAX_NEIGHBORS
        curr_err = 1
        desired_error = problem.config.TARGET_ERROR

        # Empty map
        self.neighbors_hash = {}

        confidence_level = problem.config.CONFIDENCE_LEVEL
        lower_bound: float | None = None
        upper_bound: float | None = None
        while curr_neighbors < max_neighbors and curr_err > desired_error:
            # Generate a neighbor different from all other neighbors
            nbr: Member = self.generate_neighbor(problem, curr_neighbors)
            self.neighbors.append(nbr)

            if not nbr.evaluate(problem.get_evaluator()):
                unsafe_count += 1

            curr_neighbors += 1

            # Number of evaluated members, all generated neighbors plus the original member
            evaluated = curr_neighbors + 1
            # Calculate Wilson Confidence Interval based on the estimator
            lower_bound, upper_bound = self._calculate_wilson_ci(unsafe_count / evaluated, evaluated, confidence_level)
            curr_err = (upper_bound - lower_bound) / 2.
            log.info(f'CI is now [{lower_bound:.3f},{upper_bound:.3f}] (err: +-{curr_err:.3f})')

        self.unsafe_region_probability = (lower_bound, upper_bound)

        # Fitness function 'Quality of Individual'
        ff1 = self.sparseness
        # Fitness function 'Distance to Frontier'
        p_th = problem.config.PROBABILITY_THRESHOLD
        ff2 = max(abs(upper_bound - p_th), abs(lower_bound - p_th)) / max(p_th, 1 - p_th)

        minutes, seconds = divmod(timeit.default_timer() - start, 60)
        log.info(f'Time for eval: {int(minutes):02}:{int(seconds):02}')
        log.info(f'Evaluated {self}')
        return ff1, ff2

    @classmethod
    def _calculate_wilson_ci(cls, estimator: float, sample_size: int, confidence_level: float):
        """Calculates the Wilson Confidence Interval for an estimator, given the sample size and the desired confidence
        level. Returns a tuple containing the lower and upper bounds of the CI."""
        z = cls._NORMAL_DIST.inv_cdf((1 + confidence_level) / 2.)
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

    def save(self, folder, neighborhood_image: bool = True):
        """Saves a human-interpretable representation of the individual on disk."""
        # Save a JSON representation of the individual
        json_path = folder.joinpath(self.name + '.json')
        json_path.write_text(json.dumps(self.to_dict(), indent=2))

        # Save an image of member and all neighbors
        if neighborhood_image:
            nbh_size = len(self.neighbors)

            num_cols = 3
            num_rows = math.ceil(nbh_size / num_cols) + 1
            fig = plt.figure()
            gs = fig.add_gridspec(num_rows, num_cols, hspace=0.5)
            fig.set_size_inches(15, 10)

            def plot(member: Member, pos: plt.SubplotSpec):
                ax = fig.add_subplot(pos)
                ax.set_title(f'{member}', fontsize=12)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                member.to_image(ax)

            plot(self.mbr, gs[0, 1])
            for i in range(nbh_size):
                row = math.floor(i / num_cols) + 1
                col = i % num_cols
                plot(self.neighbors[i], gs[row, col])

            fig.suptitle(f'Neighborhood size = {nbh_size}; Unsafe region = {self.unsafe_region_probability}')
            fig.savefig(folder.joinpath(self.name + '_neighborhood.png'))
            plt.close(fig)

    def to_dict(self) -> dict:
        """Serializes the individual into a dictionary that can be easily stored on disk."""
        return {'name': self.name,
                'unsafe_region': self.unsafe_region_probability,
                'archive_sparseness': self.sparseness,
                'mbr': self.mbr.to_dict(),
                'neighbors': [nbh.to_dict() for nbh in self.neighbors],
                'seed': self.seed.to_dict() if self.seed else None}

    @classmethod
    def from_dict(cls, d, individual_creator):
        """Builds an individual from its serialized representation using the provided DEAP creator."""
        mbr = T.from_dict(d['mbr'])
        neighbors = [T.from_dict(nbh) for nbh in d['neighbors']]
        seed = T.from_dict(d['seed']) if d['seed'] else None
        ind = individual_creator(mbr, seed, neighbors, d['name'])
        ind.unsafe_region_probability = d['unsafe_region']
        ind.sparseness = d['archive_sparseness']
        return ind

    def __str__(self):
        unsafe_eval = '[na]'
        if self.unsafe_region_probability:
            lb = f'{self.unsafe_region_probability[0]:.3f}'
            ub = f'{self.unsafe_region_probability[1]:.3f}'

            if self.unsafe_region_probability[0] >= 0:
                lb = '+' + lb
            if self.unsafe_region_probability[1] >= 0:
                ub = '+' + ub

            unsafe_eval = f'[{lb},{ub}]'
        # frontier_eval = frontier_eval.ljust(9)
        return (f'{self.name.ljust(6)} mbr[{self.mbr}] nbh[n={len(self.neighbors)}; '
                f'sr={len(list(filter(lambda n: n.satisfy_requirements, self.neighbors)))}] seed[{self.seed}] '
                f'ur{unsafe_eval}')
