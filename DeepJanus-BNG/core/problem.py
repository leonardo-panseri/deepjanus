from typing import Callable

from core.config import Config
from core.archive import Archive
from core.evaluator import Evaluator
from core.individual import Individual
from core.log import get_logger
from core.member import Member
from core.mutator import Mutator
from core.seed_pool import SeedPoolFolder, SeedPoolRandom

log = get_logger(__file__)


class Problem:
    """Class representing a problem to be solved by DeepJanus."""
    def __init__(self, config: Config, archive: Archive):
        self.config: Config = config
        self.archive = archive

        if self.config.SEED_POOL_STRATEGY == self.config.GEN_RANDOM:
            self.seed_pool = SeedPoolRandom(self, config.POP_SIZE)
        elif self.config.SEED_POOL_STRATEGY == self.config.GEN_RANDOM_SEEDED:
            self.seed_pool = SeedPoolFolder(self, False, config.SEED_FOLDER)
        elif self.config.SEED_POOL_STRATEGY == self.config.GEN_SEQUENTIAL_SEEDED:
            self.seed_pool = SeedPoolFolder(self, True, config.SEED_FOLDER)
        else:
            raise ValueError(f"Seed pool strategy {self.config.SEED_POOL_STRATEGY} is invalid")

        self.individual_creator = None

    def deap_individual_class(self):
        """Returns the class that represents individuals for this problem."""
        raise NotImplemented()

    def deap_generate_individual(self) -> Individual:
        """Generates a new individual from the seed pool."""
        raise NotImplemented()

    def deap_evaluate_individual(self, individual: Individual):
        """Evaluates an individual of this problem."""
        return individual.evaluate(self)

    def deap_mutate_individual(self, individual: Individual):
        """Mutates an individual of this problem."""
        individual.mutate(self)

    def member_class(self):
        """Returns the class that represents members for this problem."""
        raise NotImplemented()

    def generate_random_member(self, name: str = None) -> Member:
        """Generates a random member for this problem."""
        raise NotImplemented()

    def reseed(self, population: list[Individual]):
        """Repopulates by substituting individuals that are evolved from a seed that already generated
         a solution in the archive."""
        if len(self.archive) > 0:
            archived_seeds = [i.seed for i in self.archive]
            for i in range(len(population)):
                if population[i].seed in archived_seeds:
                    ind1 = self.deap_generate_individual()
                    log.info(f'reseed rem {population[i]}')
                    population[i] = ind1

    def on_iteration(self, idx, pop: list[Individual], logbook):
        """Problem-specific callback to execute actions at each iteration."""
        raise NotImplemented()

    def get_evaluator(self) -> Evaluator:
        """Returns the evaluator that implements the strategy for evaluating members."""
        raise NotImplemented()

    def get_mutator(self) -> Mutator:
        """Returns the mutator that implements the strategy for mutating members."""
        raise NotImplemented()
