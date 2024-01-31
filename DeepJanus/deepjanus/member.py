from .evaluator import Evaluator
from .log import get_logger
from .mutator import Mutator

log = get_logger(__file__)


class Member:
    """Class representing a member of an individual of the population"""

    counter = 1

    def __init__(self, name: str = None):
        self.name: str = name if name else f'mbr{str(Member.counter)}'
        if not name:
            Member.counter += 1

        self.satisfy_requirements: bool | None = None

    def clone(self, name: str = None) -> 'Member':
        """Creates a deep copy of the member. The name of the new copy can be passed as a parameter."""
        raise NotImplemented()

    def evaluate(self, evaluator: Evaluator) -> bool | None:
        """Evaluates the member and returns if it satisfies the requirements of the problem."""
        if self.needs_evaluation():
            self.satisfy_requirements = evaluator.evaluate(self)
        else:
            log.info(f'{self} is already evaluated, skipping')

        assert not self.needs_evaluation()
        return self.satisfy_requirements

    def mutate(self, mutator: Mutator):
        """Mutates the member."""
        mutator.mutate(self)
        self.clear_evaluation()

    def distance(self, o: 'Member') -> float:
        """Calculates the distance with another member."""
        raise NotImplemented()

    def needs_evaluation(self):
        """Returns True if the member needs to be evaluated, False if it has already been evaluated."""
        return self.satisfy_requirements is None

    def clear_evaluation(self):
        """Clears the results of evaluation for the member."""
        self.satisfy_requirements = None

    def to_tuple(self) -> tuple[float, float]:
        """Returns a 2D point representing the member, useful for visualization."""
        raise NotImplemented()

    def to_dict(self) -> dict:
        """Returns a serialized version of the member that can be saved to file."""
        raise NotImplemented()

    @classmethod
    def from_dict(cls, d: dict, name: str = None) -> 'Member':
        """Loads a member from a serialized representation. Parameter 'name' can be used to disable automatic naming
        mechanisms, as member names will not be serialized."""
        raise NotImplemented()

    def __eq__(self, other):
        raise NotImplemented()

    def __ne__(self, other):
        raise NotImplemented()

    def member_hash(self) -> str:
        """Returns a string identifying the member."""
        raise NotImplemented()

    def __str__(self):
        reqs_eval = '/'
        if self.satisfy_requirements is not None:
            reqs_eval = 'Y' if self.satisfy_requirements else 'N'
        h = self.member_hash()[-5:]
        return f'{self.name.ljust(7)} h={h} r={reqs_eval}'
