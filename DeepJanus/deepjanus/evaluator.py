# Workaround for keeping type hinting while avoiding circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .member import Member


class Evaluator:
    """Base class for implementing strategies to evaluate members"""

    def evaluate(self, member: 'Member') -> bool:
        """Evaluates a member and returns if it satisfies the requirements of the problem."""
        raise NotImplemented()
