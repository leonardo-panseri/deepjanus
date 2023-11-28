import hashlib

from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from core.member import Member
from self_driving.beamng_roads import RoadPoints, RoadBoundingBox, RoadPolygon
from self_driving.utils import RoadNodes
from self_driving.edit_distance_polyline import iterative_levenshtein


class BeamNGMember(Member):
    """Member for DeepJanus-BNG"""

    counter = 1

    def __init__(self, control_nodes: RoadNodes, sample_nodes: RoadNodes, num_spline_nodes: int,
                 road_bbox: RoadBoundingBox, name: str = None):
        """Creates a DeepJanus-BNG member. Parameter 'name' can be passed to create clones of existing members,
        disabling the automatic incremental names."""
        super().__init__(name if name else f'mbr{str(BeamNGMember.counter)}')
        if not name:
            BeamNGMember.counter += 1

        self.num_spline_nodes = num_spline_nodes
        self.control_nodes = control_nodes
        self.sample_nodes = sample_nodes
        self.road_bbox = road_bbox

    def is_valid(self):
        """Checks if the road represented by this member has a valid shape."""
        return (RoadPolygon.from_nodes(self.sample_nodes).is_valid() and
                self.road_bbox.contains(RoadPolygon.from_nodes(self.control_nodes[1:-1])))

    def clone(self):
        # Do not pass self.name, as we use this to create the offspring
        res = BeamNGMember(list(self.control_nodes), list(self.sample_nodes), self.num_spline_nodes,
                           RoadBoundingBox(self.road_bbox.bbox.bounds))

        res.distance_to_frontier = self.distance_to_frontier
        return res

    def distance(self, other: 'BeamNGMember'):
        return iterative_levenshtein(self.sample_nodes, other.sample_nodes)

    def to_tuple(self) -> tuple[float, float]:
        import numpy as np
        barycenter = np.mean(np.asarray(self.control_nodes), axis=0)[:2]
        return barycenter

    def to_dict(self) -> dict:
        return {
            'control_nodes': self.control_nodes,
            'sample_nodes': self.sample_nodes,
            'num_spline_nodes': self.num_spline_nodes,
            'road_bbox_size': self.road_bbox.bbox.bounds,
            'distance_to_frontier': self.distance_to_frontier
        }

    @classmethod
    def from_dict(cls, d: dict, name: str = None) -> 'BeamNGMember':
        road_bbox = RoadBoundingBox(d['road_bbox_size'])
        res = BeamNGMember([tuple(t) for t in d['control_nodes']],
                           [tuple(t) for t in d['sample_nodes']],
                           d['num_spline_nodes'], road_bbox, name)
        res.distance_to_frontier = d['distance_to_frontier']
        return res

    def __eq__(self, other):
        if isinstance(other, BeamNGMember):
            return self.control_nodes == other.control_nodes
        return False

    def __ne__(self, other):
        if isinstance(other, BeamNGMember):
            return self.control_nodes != other.control_nodes
        return True

    def member_hash(self):
        return hashlib.sha256(str([tuple(node) for node in self.control_nodes]).encode('UTF-8')).hexdigest()

    def to_image(self, ax: Axis = None):
        """Draws an image  of the"""
        # TODO move this after refactoring roads
        fig: Figure | None = None
        if not ax:
            fig, ax = plt.subplots()
        RoadPoints().add_middle_nodes(self.sample_nodes).plot_on_ax(ax)
        ax.axis('equal')
        return fig, ax

