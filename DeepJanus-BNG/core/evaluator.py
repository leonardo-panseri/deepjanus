# Workaround for keeping type hinting while avoiding circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.member import Member


class Evaluator:
    """Base class for implementing strategies to evaluate members"""

    def evaluate(self, member: 'Member') -> None:
        """Evaluates a member and prepares it for fitness functions calculation."""
        raise NotImplemented()
