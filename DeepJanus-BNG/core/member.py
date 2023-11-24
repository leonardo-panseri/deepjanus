class Member:
    """Class representing a member of the population"""
    def __init__(self):
        self.distance_to_boundary: float | None = None

    def clone(self):
        """Creates a deep copy of the member."""
        raise NotImplemented()

    def evaluate(self):
        """Evaluates the member."""
        raise NotImplemented()

    def mutate(self):
        """Mutates the member."""
        raise NotImplemented()

    def distance(self, o: 'Member'):
        """Calculates the distance with another member."""
        raise NotImplemented()

    def to_tuple(self):
        """Returns a 2D point representing the member, useful for visualization."""
        raise NotImplemented()

    def to_dict(self) -> dict:
        """Returns a serialized version of the member that can be saved to file."""
        raise NotImplemented()

    @classmethod
    def from_dict(cls, d: dict):
        """Loads a member from a serialized representation."""
        raise NotImplemented()
