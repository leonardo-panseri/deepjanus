import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from core.individual import Individual
from core.log import get_logger
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_wrappers import RoadPoints

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

    def save(self, folder: Path):
        # Save a JSON representation of the individual
        json_path = folder.joinpath(self.name + '.json')
        json_path.write_text(json.dumps(self.to_dict()))

        # Save an image of both the member roads
        fig, (left, right) = plt.subplots(ncols=2)
        fig.set_size_inches(15, 10)
        ml, mr = self.members_by_distance_to_boundary()

        def plot(member: BeamNGMember, ax):
            ax.set_title(f'dist to bound ~ {np.round(member.distance_to_frontier, 2)}', fontsize=12)
            road_points = RoadPoints.from_nodes(member.sample_nodes)
            road_points.plot_on_ax(ax)

        plot(ml, left)
        plot(mr, right)
        fig.suptitle(f'members distance = {self.members_distance} ; frontier distance = {self.distance_to_frontier}')
        fig.savefig(folder.joinpath(self.name + '_both_roads.svg'))
        plt.close(fig)

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
