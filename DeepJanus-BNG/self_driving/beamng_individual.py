import numpy as np

from core.individual import Individual
from core.log import get_logger
from self_driving.beamng_member import BeamNGMember

log = get_logger(__file__)


class BeamNGIndividual(Individual[BeamNGMember]):
    counter = 1

    def __init__(self, m1: BeamNGMember, m2: BeamNGMember, seed: BeamNGMember = None, name: str = None):
        """Creates a DeepJanus-BNG individual. Parameter 'name' can be passed to create clones of existing individuals,
        disabling the automatic incremental names."""
        super().__init__(name if name else f'ind{str(BeamNGIndividual.counter)}', m1, m2, seed)
        if not name:
            BeamNGIndividual.counter += 1

    def clone(self, individual_creator) -> 'BeamNGIndividual':
        # Need to use the DEAP creator to instantiate new individual
        # Do not pass self.name, as we use this to create the offspring
        res: BeamNGIndividual = individual_creator(self.m1.clone(), self.m2.clone(), self.seed)
        log.info(f'cloned to {res} from {self}')
        return res

    def semantic_distance(self, i2: 'BeamNGIndividual') -> float:
        """Calculates the distance with another individual exploiting semantic information.
        This returns the average of distances of members on the same side of the frontier."""
        i1_pos, i1_neg = self.members_by_sign()
        i2_pos, i2_neg = i2.members_by_sign()

        return np.mean([i1_pos.distance(i2_pos), i1_neg.distance(i2_neg)])

    def to_dict(self) -> dict:
        return {'name': self.name,
                'members_distance': self.members_distance,
                'm1': self.m1.to_dict(),
                'm2': self.m2.to_dict(),
                'seed': self.seed.to_dict() if self.seed else None}

    @classmethod
    def from_dict(cls, d: dict, individual_creator) -> 'BeamNGIndividual':
        m1 = BeamNGMember.from_dict(d['m1'])
        m2 = BeamNGMember.from_dict(d['m2'])
        seed = BeamNGMember.from_dict(d['seed']) if d['seed'] else None
        ind: BeamNGIndividual = individual_creator(m1, m2, seed, d['name'])
        ind.members_distance = d['members_distance']
        return ind
