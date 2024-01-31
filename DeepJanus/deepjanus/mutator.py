# Workaround for keeping type hinting while avoiding circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .member import Member


class Mutator:
    """Base class for implementing mutation strategies for members"""

    def mutate(self, member: 'Member'):
        """Mutates a member."""
        raise NotImplemented()
