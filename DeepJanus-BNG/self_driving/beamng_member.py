import hashlib

from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from core.member import Member
from self_driving.beamng_roads import BeamNGRoad
from self_driving.shapely_roads import RoadGenerationBoundary, RoadPolygon
from self_driving.utils import Point4D
from self_driving.levenshtein_distance import iterative_levenshtein


class BeamNGMember(Member):
    """Member for DeepJanus-BNG"""

    counter = 1

    def __init__(self, control_nodes: list[Point4D], sample_nodes: list[Point4D], num_spline_nodes: int,
                 generation_boundary: RoadGenerationBoundary, name: str = None):
        """Creates a DeepJanus-BNG member. Parameter 'name' can be passed to create clones of existing members,
        disabling the automatic incremental names."""
        super().__init__(name if name else f'mbr{str(BeamNGMember.counter)}')
        if not name:
            BeamNGMember.counter += 1

        self.road = BeamNGRoad(sample_nodes, control_nodes, num_spline_nodes)
        self.generation_boundary = generation_boundary

    def is_valid(self):
        """Checks if the road represented by this member has a valid shape and is inside the generation boundary."""
        return (RoadPolygon(self.road).is_valid() and
                self.generation_boundary.contains(RoadPolygon.from_nodes(self.road.control_nodes[1:-1])))

    def clone(self):
        # Do not pass self.name, as we use this to create the offspring
        res = BeamNGMember(list(self.road.control_nodes), list(self.road.nodes), self.road.num_spline_nodes,
                           RoadGenerationBoundary(self.generation_boundary.bbox.bounds))

        res.distance_to_frontier = self.distance_to_frontier
        return res

    def distance(self, other: 'BeamNGMember'):
        return iterative_levenshtein(self.road.nodes, other.road.nodes)

    def to_tuple(self) -> tuple[float, float]:
        import numpy as np
        barycenter = np.mean(np.asarray(self.road.control_nodes), axis=0)[:2]
        return barycenter

    def to_dict(self) -> dict:
        return {
            'control_nodes': self.road.control_nodes,
            'sample_nodes': self.road.nodes,
            'num_spline_nodes': self.road.num_spline_nodes,
            'road_bbox_size': self.generation_boundary.bbox.bounds,
            'distance_to_frontier': self.distance_to_frontier
        }

    @classmethod
    def from_dict(cls, d: dict, name: str = None) -> 'BeamNGMember':
        road_bbox = RoadGenerationBoundary(d['road_bbox_size'])
        res = BeamNGMember([tuple(t) for t in d['control_nodes']],
                           [tuple(t) for t in d['sample_nodes']],
                           d['num_spline_nodes'], road_bbox, name)
        res.distance_to_frontier = d['distance_to_frontier']
        return res

    def __eq__(self, other):
        if isinstance(other, BeamNGMember):
            return self.road.control_nodes == other.road.control_nodes
        return False

    def __ne__(self, other):
        if isinstance(other, BeamNGMember):
            return self.road.control_nodes != other.road.control_nodes
        return True

    def member_hash(self):
        return hashlib.sha256(str([tuple(node) for node in self.road.control_nodes]).encode('UTF-8')).hexdigest()

    def to_image(self, ax: Axis = None):
        """Plots the shape of the road for the member in a matplotlib figure."""
        fig: Figure | None = None
        if not ax:
            fig, ax = plt.subplots()
        self.road.to_image(ax)
        return fig, ax
