import json
import math
from pathlib import Path

from matplotlib import pyplot as plt

from core.individual import Individual
from core.log import get_logger
from self_driving.beamng_member import BeamNGMember

log = get_logger(__file__)


class BeamNGIndividual(Individual[BeamNGMember]):
    """Individual for DeepJanus-BNG"""

    def __init__(self, mbr: BeamNGMember, seed: BeamNGMember = None,
                 neighbors: list[BeamNGMember] = None, name: str = None):
        """Creates a DeepJanus-BNG individual. Parameter 'name' can be passed to create clones of existing individuals,
        disabling the automatic incremental names."""
        super().__init__(mbr, seed, neighbors, name)

    def clone(self, individual_creator) -> 'BeamNGIndividual':
        # Need to use the DEAP creator to instantiate new individual
        # Do not pass self.name, as we use this to create the offspring
        res: BeamNGIndividual = individual_creator(self.mbr.clone(), self.seed)
        log.info(f'Cloned to {res} from {self}')
        return res

    def save(self, folder: Path):
        # Save a JSON representation of the individual
        json_path = folder.joinpath(self.name + '.json')
        json_path.write_text(json.dumps(self.to_dict()))

        nbh_size = len(self.neighbors)

        # Save an image of member road and all neighbors
        num_cols = 3
        num_rows = math.ceil(nbh_size / num_cols) + 1
        fig = plt.figure()
        gs = fig.add_gridspec(num_rows, num_cols)
        fig.set_size_inches(15, 10)

        def plot(member: BeamNGMember, pos: plt.SubplotSpec):
            ax = fig.add_subplot(pos)
            ax.set_title(f'{member}', fontsize=12)
            member.to_image(ax)

        plot(self.mbr, gs[0, :])
        for i in range(nbh_size):
            row = math.floor(i / num_cols)
            col = i % num_cols
            plot(self.neighbors[i], gs[row, col])

        fig.suptitle(f'Neighborhood size = {nbh_size}; Frontier distance = {self.distance_to_frontier}')
        fig.savefig(folder.joinpath(self.name + '_neighborhood.svg'))
        plt.close(fig)

    def to_dict(self) -> dict:
        return {'name': self.name,
                'frontier_dist': self.distance_to_frontier,
                'mbr': self.mbr.to_dict(),
                'neighbors': [nbh.to_dict() for nbh in self.neighbors],
                'seed': self.seed.to_dict() if self.seed else None}

    @classmethod
    def from_dict(cls, d: dict, individual_creator) -> 'BeamNGIndividual':
        mbr = BeamNGMember.from_dict(d['mbr'])
        neighbors = [BeamNGMember.from_dict(nbh) for nbh in d['neighbors']]
        seed = BeamNGMember.from_dict(d['seed']) if d['seed'] else None
        ind: BeamNGIndividual = individual_creator(mbr, seed, neighbors, d['name'])
        ind.distance_to_frontier = d['frontier_dist']
        return ind
