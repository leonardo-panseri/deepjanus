from core.evaluator import Evaluator
from core.mutator import Mutator


class Member:
    """Class representing a member of the population"""

    def __init__(self, name: str):
        self.name: str = name

        self.distance_to_frontier: float | None = None

    def clone(self):
        """Creates a deep copy of the member."""
        raise NotImplemented()

    def evaluate(self, evaluator: Evaluator):
        """Evaluates the member."""
        if self.needs_evaluation():
            evaluator.evaluate(self)
        assert not self.needs_evaluation()

    def mutate(self, mutator: Mutator):
        """Mutates the member."""
        mutator.mutate(self)
        self.clear_evaluation()

    def distance(self, o: 'Member') -> float:
        """Calculates the distance with another member."""
        raise NotImplemented()

    def needs_evaluation(self):
        """Returns True if the member needs to be evaluated, False if it has already been evaluated."""
        return self.distance_to_frontier is None

    def clear_evaluation(self):
        """Clears the results of evaluation for the member."""
        self.distance_to_frontier = None

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

    def __repr__(self):
        frontier_eval = 'na'
        if self.distance_to_frontier:
            frontier_eval = str(self.distance_to_frontier)
            if self.distance_to_frontier > 0:
                frontier_eval = '+' + frontier_eval
            frontier_eval = '~' + frontier_eval
        frontier_eval = frontier_eval[:7].ljust(7)
        h = self.member_hash()[-5:]
        return f'{self.name.ljust(7)} h={h} f={frontier_eval}'
