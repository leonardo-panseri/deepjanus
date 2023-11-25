from core.config import Config
from core.archive import Archive
from core.individual import Individual
from core.member import Member


class Problem:
    """Class representing a problem to be solved by DeepJanus."""
    def __init__(self, config: Config, archive: Archive):
        self.config: Config = config
        self.archive = archive

    def deap_generate_individual(self) -> Individual:
        """Generates a new individual from the seed pool."""
        raise NotImplemented()

    @classmethod
    def deap_mutate_individual(cls, individual: Individual):
        """Mutates an individual of this problem."""
        individual.mutate()

    def deap_evaluate_individual(self, individual: Individual):
        """Evaluates an individual of this problem."""
        raise NotImplemented()

    def deap_individual_class(self):
        """Returns the class that represents individuals for this problem."""
        raise NotImplemented()

    def on_iteration(self, idx, pop: list[Individual], logbook):
        """Problem-specific callback to execute actions at each iteration."""
        raise NotImplemented()

    def member_class(self):
        """Returns the class that represents members for this problem."""
        raise NotImplemented()

    def reseed(self, population, offspring):
        """Repopulates by substituting individuals that are evolved from a seed that already generated
         a solution in the archive."""
        raise NotImplemented()

    def generate_random_member(self) -> Member:
        """Generates a random member for this problem."""
        raise NotImplemented()

    def pre_evaluate_members(self, individuals: list[Individual]):
        pass
