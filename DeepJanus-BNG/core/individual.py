from numpy import mean

from member import Member


class Individual:
    """Class representing an individual of the population"""
    def __init__(self, m1: Member, m2: Member):
        self.name: str
        self.m1: Member = m1
        self.m2: Member = m2
        self.members_distance: float | None = None
        self.sparseness: float | None = None
        self.distance_to_frontier: float | None = None
        self.seed: Member | None = None

    def clone(self) -> 'Individual':
        """Creates a deep copy of the individual."""
        raise NotImplemented()

    def evaluate(self) -> tuple[float, float]:
        """Evaluates the individual and returns the two fitness values."""
        raise NotImplemented()

    def mutate(self):
        """Mutates the individual."""
        raise NotImplemented()

    def distance(self, i2: 'Individual'):
        """Calculates the distance with another individual."""
        i1 = self
        a = i1.m1.distance(i2.m1)
        b = i1.m1.distance(i2.m2)
        c = i1.m2.distance(i2.m1)
        d = i1.m2.distance(i2.m2)

        dist = mean([min(a, b), min(c, d), min(a, c), min(b, d)])
        return dist

    def semantic_distance(self, i2: 'Individual'):
        """Calculates the distance with another individual exploiting semantic information."""
        raise NotImplemented()

    def members_by_sign(self) -> tuple[Member, Member]:
        """Returns a tuple containing first the member outside the frontier, and second the member inside."""
        result = self.members_by_distance_to_boundary()

        assert result[0].distance_to_boundary < 0, str(result[0].distance_to_boundary) + ' ' + str(self)
        assert result[1].distance_to_boundary >= 0, str(result[1].distance_to_boundary) + ' ' + str(self)
        return result

    def members_by_distance_to_boundary(self) -> tuple[Member, Member]:
        """Returns a tuple containing the members sorted in ascending order by their distance to the frontier."""
        msg = 'in order to use this distance metrics you need to evaluate the member'
        assert self.m1.distance_to_boundary, msg
        assert self.m2.distance_to_boundary, msg

        result = sorted([self.m1, self.m2], key=lambda m: m.distance_to_boundary)
        return tuple(result)
