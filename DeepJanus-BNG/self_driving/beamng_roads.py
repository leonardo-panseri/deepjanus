import json
import uuid

import numpy as np
from matplotlib import pyplot as plt

from core.log import get_logger
from self_driving.beamng_vehicles import BeamNGPose
from self_driving.curve_interpolation import catmull_rom
from self_driving.utils import get_node_coords, Point3D, Point4D, Point2D

log = get_logger(__file__)


class BeamNGWaypoint:
    """Represents a waypoint in a simulation scenario"""

    def __init__(self, name: str, position: Point3D):
        self.name = name
        self.position = position
        self.persistent_id = str(uuid.uuid4())

    def to_json(self) -> str:
        """Returns a JSON representation of the waypoint that will be understood by BeamNG."""
        return json.dumps({
            'name': self.name, 'class': 'BeamNGWaypoint', 'persistentId': self.persistent_id,
            '__parent': 'generated', 'position': self.position, 'scale': [4, 4, 4]})


class BeamNGRoad:
    """Represents a road in a simulation scenario"""

    NAME = 'generated_road'

    def __init__(self, nodes: list[Point4D], control_nodes: list[Point4D] = None, num_spline_nodes=20):
        assert len(nodes) >= 2, f'At least, two nodes are needed'
        assert all(len(point) == 4 for point in nodes), f'A node is a tuple of 4 elements (x,y,z,road_width)'

        self.control_nodes = control_nodes
        self.nodes = nodes
        self.num_spline_nodes = num_spline_nodes

        self.lane_marker_left: list[Point2D] = [(.0, .0)] * len(nodes)
        self.lane_marker_right: list[Point2D] = [(.0, .0)] * len(nodes)
        self.recalculate_lane_markers()

        self.waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))

        self.persistent_id = str(uuid.uuid4())

    def mutate_node(self, index: int, mutation: Point4D):
        self.control_nodes[index] = mutation
        self.nodes = catmull_rom(self.control_nodes, self.num_spline_nodes)
        self.recalculate_lane_markers()

    def recalculate_lane_markers(self):
        """Calculates the list of points representing left and right lane markers given from the interpolated nodes
        (that can be seen as center lane markers)."""
        def calculate_marker_points(curr_point: Point4D, nxt_point: Point4D) -> tuple[Point2D, Point2D]:
            """Calculates left and right marker points for a given middle point of a road. The next point is also
            needed to calculate the normal directions to the road direction."""
            curr = np.array(curr_point[0:2])
            # Vector from the current to the next point
            direction = np.subtract(nxt_point[0:2], curr)
            # Vector from the current to the next point which length is half the road width
            direction = (direction / np.linalg.norm(direction)) * curr_point[3] / 2
            # Calculate normal vectors that will identify lane marker points
            left = curr + np.array([-direction[1], direction[0]])
            right = curr + np.array([direction[1], -direction[0]])
            return tuple(left), tuple(right)

        for i in range(len(self.nodes) - 1):
            l, r = calculate_marker_points(self.nodes[i], self.nodes[i + 1])
            self.lane_marker_left[i] = l
            self.lane_marker_right[i] = r

        # Last middle point, right and left are inverted because we are going backward
        self.lane_marker_right[-1], self.lane_marker_left[-1] = calculate_marker_points(self.nodes[-1],
                                                                                        self.nodes[-2])

    def vehicle_start_pose(self, meters_from_road_start=2.5, road_point_index=0):
        """Calculates the position and rotation of a vehicle that needs to drive from the start of the road."""
        assert len(self.nodes) > road_point_index, (f'road length is {len(self.nodes)} and '
                                                    f'it does not have index {road_point_index}')
        curr = self.nodes[road_point_index]
        curr_right = self.lane_marker_right[road_point_index]
        nxt = self.nodes[road_point_index + 1]

        direction = np.subtract(nxt[0:2], curr[0:2])
        direction = (direction / np.linalg.norm(direction)) * meters_from_road_start

        origin = np.add(curr[0:2], curr_right[0:2]) / 2
        deg = np.degrees(np.arctan2([-direction[0]], [-direction[1]]))
        return BeamNGPose(pos=tuple(origin + direction) + (curr[2],), rot=(0, 0, deg[0]))

    def to_image(self, ax: plt.Axes):
        """Plots the shape of the road in a matplotlib figure."""
        def plot_xy(points: list[tuple], color: str, line_width: float):
            tup = list(zip(*points))
            ax.plot(tup[0], tup[1], color=color, linewidth=line_width)

        ax.set_facecolor('#7D9051')  # green
        plot_xy(self.nodes, '#FEA952', line_width=1)  # orange
        plot_xy(self.lane_marker_left, 'white', line_width=1)
        plot_xy(self.lane_marker_right, 'white', line_width=1)
        ax.axis('equal')

    def to_json(self) -> str:
        """Returns a JSON representation of the road that will be understood by BeamNG."""
        assert len(self.nodes) > 0, 'there are no points in this road'
        return json.dumps({
            'name': self.NAME, 'class': 'DecalRoad', 'breakAngle': 180, 'distanceFade': [1000, 1000],
            'drivability': 1, 'material': 'tig_road_rubber_sticky', 'overObjects': True,
            'persistentId': self.persistent_id, '__parent': 'generated',
            'position': tuple(self.nodes[0][:3]), 'textureLength': 2.5, 'nodes': self.nodes})

    def to_dict(self) -> dict:
        """Returns a dict representing the road."""
        return {
            'name': self.NAME,
            'nodes': self.nodes
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Creates a road from its dict representation."""
        return BeamNGRoad(d['nodes'])
