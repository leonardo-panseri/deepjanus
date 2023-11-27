import random
import timeit
from typing import TypeVar, Generic

from numpy import mean

from core.log import get_logger
from core.member import Member

# Workaround for keeping type hinting while avoiding circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.problem import Problem

log = get_logger(__file__)
T = TypeVar('T', bound=Member)


class Individual(Generic[T]):
    """Class representing an individual of the population"""

    def __init__(self, name: str, m1: T, m2: T, seed: T = None):
        self.name: str = name

        self.m1: T = m1
        self.m2: T = m2
        self.seed: T | None = seed

        self.members_distance: float | None = None

        self.sparseness: float | None = None
        self.distance_to_frontier: float | None = None

    def clone(self, individual_creator) -> 'Individual':
        """Creates a deep copy of the individual using the provided DEAP creator."""
        raise NotImplemented()

    def evaluate(self, problem: 'Problem') -> tuple[float, float]:
        """Evaluates the individual and returns the two fitness values."""
        assert self.m1 != self.m2
        start = timeit.default_timer()

        self.members_distance = self.m1.distance(self.m2)
        stop = timeit.default_timer()
        log.info(f'Time to mem dist: {stop - start}')

        self.sparseness = problem.archive.evaluate_sparseness(self)
        stop = timeit.default_timer()
        log.info(f'Time to sparseness: {stop - start}; archive len: {len(problem.archive)}')

        self.m1.evaluate(problem.get_evaluator())
        self.m2.evaluate(problem.get_evaluator())
        stop = timeit.default_timer()
        log.info(f'Time to eval: {stop - start}')

        # Fitness function 'Quality of Individual'
        ff1 = self.sparseness - (problem.config.K_SD * self.members_distance)
        # Fitness function 'Closeness to Frontier'
        distance_to_frontier = self.m1.distance_to_frontier * self.m2.distance_to_frontier
        self.distance_to_frontier = distance_to_frontier if distance_to_frontier > 0 else -0.1

        stop = timeit.default_timer()
        log.info(f'Total Time: {stop - start}')
        log.info(f'evaluated {self}')
        return ff1, self.distance_to_frontier

    def mutate(self, problem: 'Problem'):
        """Mutates the individual."""
        member_to_mutate = self.m1 if random.randrange(2) == 0 else self.m2
        members_not_equal = False
        while not members_not_equal:
            member_to_mutate.mutate(problem.get_mutator())
            if self.m1 != self.m2:
                members_not_equal = True
        self.members_distance = None
        log.info(f'mutated {member_to_mutate}')

    def distance(self, i2: 'Individual') -> float:
        """Calculates the distance with another individual."""
        i1 = self
        a = i1.m1.distance(i2.m1)
        b = i1.m1.distance(i2.m2)
        c = i1.m2.distance(i2.m1)
        d = i1.m2.distance(i2.m2)

        dist = mean([min(a, b), min(c, d), min(a, c), min(b, d)])
        return dist

    def semantic_distance(self, i2: 'Individual') -> float:
        """Calculates the distance with another individual exploiting semantic information."""
        raise NotImplemented()

    def members_by_sign(self) -> tuple[T, T]:
        """Returns a tuple containing first the member outside the frontier, and second the member inside."""
        result = self.members_by_distance_to_boundary()

        assert result[0].distance_to_frontier < 0, str(result[0].distance_to_frontier) + ' ' + str(self)
        assert result[1].distance_to_frontier >= 0, str(result[1].distance_to_frontier) + ' ' + str(self)
        return result

    def members_by_distance_to_boundary(self) -> tuple[T, T]:
        """Returns a tuple containing the members sorted in ascending order by their distance to the frontier."""
        msg = 'in order to use this distance metrics you need to evaluate the member'
        assert self.m1.distance_to_frontier, msg
        assert self.m2.distance_to_frontier, msg

        result = sorted([self.m1, self.m2], key=lambda m: m.distance_to_frontier)
        return tuple(result)

    def to_dict(self) -> dict:
        """Serializes the individual into a dictionary that can be easily stored on disk."""
        raise NotImplemented()

    @classmethod
    def from_dict(cls, d, individual_creator) -> 'Individual':
        """Builds an individual from its serialized representation using the provided DEAP creator."""
        raise NotImplemented()

    def __str__(self):
        dist = str(self.members_distance).ljust(4)[:4]
        return f'{self.name.ljust(6)} dist={dist} m1[{self.m1}] m2[{self.m2}] seed[{self.seed}]'
