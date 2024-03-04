import json
import math
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

    def __init__(self, mbr: T, seed_index: int = None, neighbors: list[T] = None, name: str = None):
        """Creates a DeepJanus individual. Parameter 'name' can be passed to create clones of existing individuals,
        disabling the automatic incremental names."""
        self.name: str = name if name else f'ind{str(Individual.counter)}'
        if not name:
            Individual.counter += 1

        self.mbr: T = mbr
        self.neighbors: list[T] = neighbors if neighbors else []
        self.seed_index: int | None = seed_index

        self.sparseness: float | None = None
        self.unsafe_region_probability: tuple[float, float] | None = None

        # Map to quickly check if neighbors are all different from each other
        self.neighbors_hash = {}

    def clone(self, individual_creator):
        """Creates a deep copy of the individual using the provided DEAP creator."""
        # Need to use the DEAP creator to instantiate new individual
        # Do not pass self.name, as we use this to create the offspring
        res = individual_creator(self.mbr.clone(), self.seed_index)
        log.info(f'Cloned to {res} from {self}')
        return res

    def generate_neighbor(self, problem: 'Problem') -> Member:
        """Generate a neighbor of this individual different from all other neighbors already generated."""
        equal_nbr = True
        nbr: Member | None = None
        while equal_nbr:
            nbr = self.mbr.clone(f'nbr{len(self.neighbors_hash) + 1}_{self.name.replace("ind", "")}')
            nbr.mutate(problem.get_mutator())
            equal_nbr = nbr.member_hash() in self.neighbors_hash
        self.neighbors_hash[nbr.member_hash()] = True
        return nbr

    def evaluate(self, problem: 'Problem') -> tuple[float, float]:
        """Evaluates the individual and returns the two fitness values."""
        return problem.get_evaluator().evaluate_individual(self, problem)

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
        json_path.write_text(json.dumps(self.to_dict()))

        # Save an image of member and all neighbors
        if neighborhood_image:
            nbh_size = len(self.neighbors)

            num_cols = 3
            num_rows = math.ceil(nbh_size / num_cols) + 1
            fig = plt.figure()
            gs = fig.add_gridspec(num_rows, num_cols, hspace=0.8)
            fig.set_size_inches(20, 15)

            def plot(member: Member, pos: plt.SubplotSpec):
                ax = fig.add_subplot(pos)
                ax.set_title(f'{member}', fontsize=12)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                member.to_image(ax)

            plot(self.mbr, gs[0, 1])
            satisfy_requirements = 0
            for i in range(nbh_size):
                row = math.floor(i / num_cols) + 1
                col = i % num_cols
                plot(self.neighbors[i], gs[row, col])
                if self.neighbors[i].satisfy_requirements:
                    satisfy_requirements += 1

            fig.suptitle(f'Neighbors satisfying req. = {satisfy_requirements}/{nbh_size}; '
                         f'Unsafe region = {self.unsafe_region_probability[0]:.3f},{self.unsafe_region_probability[1]:.3f}')
            fig.savefig(folder.joinpath(self.name + '_neighborhood.png'), bbox_inches='tight')
            plt.close(fig)

    def to_dict(self) -> dict:
        """Serializes the individual into a dictionary that can be easily stored on disk."""
        return {'name': self.name,
                'unsafe_region': self.unsafe_region_probability,
                'archive_sparseness': self.sparseness,
                'mbr': self.mbr.to_dict(),
                'neighbors': [nbh.to_dict() for nbh in self.neighbors],
                'seed': self.seed_index}

    @classmethod
    def from_dict(cls, d, individual_creator):
        """Builds an individual from its serialized representation using the provided DEAP creator."""
        mbr = T.from_dict(d['mbr'])
        neighbors = [T.from_dict(nbh) for nbh in d['neighbors']]
        seed_index = d['seed']
        ind = individual_creator(mbr, seed_index, neighbors, d['name'])
        ind.unsafe_region_probability = d['unsafe_region']
        ind.sparseness = d['archive_sparseness']
        return ind

    def __str__(self):
        unsafe_eval = 'na'
        if self.unsafe_region_probability:
            lb = f'{self.unsafe_region_probability[0]:.3f}'
            ub = f'{self.unsafe_region_probability[1]:.3f}'

            if self.unsafe_region_probability[0] >= 0:
                lb = '+' + lb
            if self.unsafe_region_probability[1] >= 0:
                ub = '+' + ub

            unsafe_eval = f'{lb},{ub}'
        # frontier_eval = frontier_eval.ljust(9)
        f = 'na'
        if self.fitness.values:
            f = f'{self.fitness.values[0]:.3f},{self.fitness.values[1]:.3f}'
        return (f'{self.name.ljust(6)} mbr[{self.mbr}] nbh[n={len(self.neighbors)}; '
                f'sr={len(list(filter(lambda n: n.satisfy_requirements, self.neighbors)))}] seed[{self.seed_index}] '
                f'ur[{unsafe_eval}] f[{f}]')
