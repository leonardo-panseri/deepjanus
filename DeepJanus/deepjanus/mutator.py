from random import getrandbits, randint, uniform
from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    from .member import Member


class Mutator:
    """Base class for implementing mutation strategies for members"""

    def __init__(self, config: Config):
        self.lower_bound = config.MUTATION_LOWER_BOUND
        self.upper_bound = config.MUTATION_UPPER_BOUND
        self.randomize_sign = config.MUTATION_RANDOMIZE_SIGN

    def mutate(self, member: 'Member'):
        """Mutates a member."""
        raise NotImplementedError()

    def _check_types(self, t: type):
        """Checks if mutation bounds are both of the given type."""
        return isinstance(self.lower_bound, t) and isinstance(self.upper_bound, t)

    def get_random_mutation_extent(self):
        """Gets a random mutation extent inside bounds set in the config.
        If MUTATION_RANDOMIZE_SIGN is True the result can have both plus or minus sign."""
        if self._check_types(int):
            mutation_extent = randint(self.lower_bound, self.upper_bound)
        elif self._check_types(float):
            mutation_extent = uniform(self.lower_bound, self.upper_bound)
        else:
            raise TypeError('Mutation lower and upper bound must be both int or both float')

        if self.randomize_sign:
            if getrandbits(1):
                mutation_extent = -mutation_extent

        return mutation_extent
