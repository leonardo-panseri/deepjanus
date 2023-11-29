from random import randint
import math

from shapely import Point, Polygon, LineString, box
import numpy as np

from self_driving.beamng_roads import BeamNGRoad, log
from self_driving.curve_interpolation import catmull_rom
from self_driving.points import Point4D, Point2D


class RoadPolygon:
    """Represents the road as a Shapely geometrical object (a polygon or a sequence of polygons)."""

    @classmethod
    def from_nodes(cls, nodes: list[Point4D]):
        """Builds a road polygon from road nodes."""
        return RoadPolygon(BeamNGRoad(nodes))

    def __init__(self, road: BeamNGRoad):
        assert len(road.lane_marker_left) == len(road.lane_marker_right) == len(road.nodes)
        assert len(road.nodes) >= 2
        assert all(x[3] == road.nodes[0][3] for x in road.nodes), \
            "The width of the road should be equal everywhere."
        self.road = road
        self.road_width = road.nodes[0][3]
        self.polygons = self._compute_polygons()
        self.polygon = self._compute_polygon()
        self.right_polygon = self._compute_right_polygon()
        self.left_polygon = self._compute_left_polygon()
        self.polyline = self._compute_polyline()
        self.right_polyline = self._compute_right_polyline()
        self.left_polyline = self._compute_left_polyline()
        self.num_polygons = len(self.polygons)

    def _compute_polygons(self) -> list[Polygon]:
        """Creates a list of Polygon objects that represent the road.
        Each polygon represents a segment of the road. Two objects adjacent in
        the returned list represent adjacent segments of the road."""
        polygons = []
        for left, right, left1, right1, in zip(self.road.lane_marker_left,
                                               self.road.lane_marker_right,
                                               self.road.lane_marker_left[1:],
                                               self.road.lane_marker_right[1:]):
            assert len(left) >= 2 and len(right) >= 2 and len(left1) >= 2 and len(right1) >= 2
            # Ignore the z coordinate.
            polygons.append(Polygon([left[:2], left1[:2], right1[:2], right[:2]]))
        return polygons

    def _compute_polygon(self) -> Polygon:
        """Returns a single polygon that represents the whole road."""
        road_poly = self.road.lane_marker_left.copy()
        road_poly.extend(self.road.lane_marker_right[::-1])
        return Polygon(road_poly)

    def _compute_right_polygon(self) -> Polygon:
        """Returns a single polygon that represents the right lane of the road."""
        road_poly = [(p[0], p[1]) for p in self.road.nodes]
        road_poly.extend(self.road.lane_marker_right[::-1])
        return Polygon(road_poly)

    def _compute_left_polygon(self) -> Polygon:
        """Returns a single polygon that represents the left lane of the road."""
        road_poly = self.road.lane_marker_left.copy()
        road_poly.extend([(p[0], p[1]) for p in self.road.nodes][::-1])
        return Polygon(road_poly)

    def _compute_polyline(self) -> LineString:
        """Computes a LineString representing the polyline of the spin (or middle) of the road."""
        return LineString([(n[0], n[1]) for n in self.road.nodes])

    def _compute_right_polyline(self) -> LineString:
        """Computes a LineString representing the polyline of the spin (or middle) of the right lane of the road."""
        return LineString([((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in
                           zip(self.road.nodes, self.road.lane_marker_right)])

    def _compute_left_polyline(self) -> LineString:
        """Computes a LineString representing the polyline of the spin (or middle) of the left lane of the road."""
        return LineString([((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in
                           zip(self.road.lane_marker_left, self.road.nodes)])

    def _get_neighbouring_polygons(self, i: int) -> list[int]:
        """Returns the indices of the neighbouring polygons of the polygon with index i."""
        if self.num_polygons == 1:
            assert i == 0
            return []
        assert 0 <= i < self.num_polygons
        if i == 0:
            return [i + 1]
        elif i == self.num_polygons - 1:
            return [i - 1]
        else:
            assert self.num_polygons >= 3
            return [i - 1, i + 1]

    def _are_neighbouring_polygons(self, i: int, j: int) -> bool:
        """Returns true if the polygons represented by the indices i and j are adjacent."""
        return j in self._get_neighbouring_polygons(i)

    def is_valid(self) -> bool:
        """Returns true if the current RoadPolygon representation of the road is valid,
        that is, if there are no intersections between non-adjacent polygons and if
        the adjacent polygons have as intersection a LineString (a line or segment)."""
        if self.num_polygons == 0:
            log.debug("No polygon constructed.")
            return False

        for i, polygon in enumerate(self.polygons):
            if not polygon.is_valid:
                log.debug("Polygon %s is invalid." % polygon)
                return False

        for i, polygon in enumerate(self.polygons):
            for j, other in enumerate(self.polygons):
                # Ignore the case when other is equal to the polygon.
                if other == polygon:
                    assert i == j
                    continue
                if polygon.contains(other) or other.contains(polygon):
                    log.debug("No polygon should contain any other polygon.")
                    return False
                if not self._are_neighbouring_polygons(i, j) and other.intersects(polygon):
                    log.debug("The non-neighbouring polygons %s and %s intersect." % (polygon, other))
                    return False
                if self._are_neighbouring_polygons(i, j) and not isinstance(other.intersection(polygon), LineString):
                    log.debug("The neighbouring polygons %s and %s have an intersection of type %s." % (
                        polygon, other, type(other.intersection(polygon))))
                    return False
        log.debug("The road is apparently valid.")
        return True


class RoadGenerationBoundary:
    """Rectangle delimiting the valid road-generation area"""

    def __init__(self, bbox_size: tuple[float, float, float, float]):
        assert len(bbox_size) == 4
        self.bbox = box(*bbox_size)

    def _get_vertices(self) -> list[Point]:
        xs, ys = self.bbox.exterior.coords.xy
        xys = list(zip(xs, ys))
        return [Point(xy) for xy in xys]

    def _get_sides(self) -> list[LineString]:
        sides = []
        xs, ys = self.bbox.exterior.coords.xy
        xys = list(zip(xs, ys))
        for p1, p2 in zip(xys[:-1], xys[1:]):
            sides.append(LineString([p1, p2]))
        return sides

    def intersects_sides(self, point: Point) -> bool:
        """Checks if a point is on the sides of the rectangle."""
        for side in self._get_sides():
            if side.intersects(point):
                return True
        return False

    def intersects_vertices(self, point: Point) -> bool:
        """Checks if a point is on the vertices of the rectangle."""
        for vertex in self._get_vertices():
            if vertex.intersects(point):
                return True
        return False

    def intersects_boundary(self, other: Polygon) -> bool:
        """Checks if a polygon intersects with the boundary of the rectangle."""
        return other.intersects(self.bbox.boundary)

    def contains(self, other: RoadPolygon) -> bool:
        """Checks if the rectangle contains a polygon."""
        return self.bbox.contains(other.polyline)


class RoadGenerator:
    """Random road generator"""

    MAX_ANGLE = 80
    NUM_SPLINE_NODES = 20
    NUM_INITIAL_SEGMENTS_THRESHOLD = 2
    NUM_UNDO_ATTEMPTS = 20
    SEG_LENGTH = 25

    def __init__(self, num_control_nodes=15, max_angle=MAX_ANGLE, seg_length=SEG_LENGTH,
                 num_spline_nodes=NUM_SPLINE_NODES, initial_node=(0.0, 0.0, -28.0, 8.0),
                 generation_boundary=(-250, 0, 250, 500)):
        assert num_control_nodes > 1 and num_spline_nodes > 0
        assert 0 <= max_angle <= 360
        assert seg_length > 0
        assert len(initial_node) == 4 and len(generation_boundary) == 4
        self.num_control_nodes = num_control_nodes
        self.num_spline_nodes = num_spline_nodes
        self.initial_node = initial_node
        self.max_angle = max_angle
        self.seg_length = seg_length
        self.generation_boundary = RoadGenerationBoundary(generation_boundary)
        assert not self.generation_boundary.intersects_vertices(self._get_initial_point())
        assert self.generation_boundary.intersects_sides(self._get_initial_point())

    def generate_control_nodes(self, attempts=NUM_UNDO_ATTEMPTS) -> list[Point4D]:
        """Generates control nodes that can be interpolated to obtain the shape of the road.
        Note that this will return the control nodes plus the two extra nodes needed by the current Catmull-Rom model.
        These two extra nodes will be the first and the last of the returned nodes."""
        nodes = []
        # The road generation ends when there are the control nodes plus the two extra nodes
        # needed by the current Catmull-Rom model
        while len(nodes) - 2 != self.num_control_nodes:
            nodes = [self._get_initial_control_node(), self.initial_node]

            # Number of valid generated control nodes
            valid = 0

            # When attempt >= attempts and the skeleton of the road is still invalid,
            # the construction of the skeleton starts again from the beginning.
            # attempt is incremented every time the skeleton is invalid.
            attempt = 0

            while valid < self.num_control_nodes and attempt <= attempts:
                nodes.append(self._get_next_node(nodes[-2], nodes[-1], self._get_next_max_angle(valid)))
                road_polygon = RoadPolygon.from_nodes(nodes)

                # Number of iterations used to attempt to add a valid next control node
                # before also removing the previous control node.
                budget = self.num_control_nodes - valid
                assert budget >= 1

                intersect_boundary = self.generation_boundary.intersects_boundary(road_polygon.polygons[-1])
                is_valid = road_polygon.is_valid() and (
                        ((valid == 0) and intersect_boundary) or ((valid > 0) and not intersect_boundary))
                while not is_valid and budget > 0:
                    nodes.pop()
                    budget -= 1
                    attempt += 1

                    nodes.append(self._get_next_node(nodes[-2], nodes[-1], self._get_next_max_angle(valid)))
                    road_polygon = RoadPolygon.from_nodes(nodes)

                    intersect_boundary = self.generation_boundary.intersects_boundary(road_polygon.polygons[-1])
                    is_valid = road_polygon.is_valid() and (
                                ((valid == 0) and intersect_boundary) or ((valid > 0) and not intersect_boundary))
                    # if visualise:
                    #     fig = plot_road_bbox(self.road_bbox)
                    #     plot_road_polygon(road_polygon, title="RoadPolygon i=%s" % i, fig=fig)

                if is_valid:
                    valid += 1
                else:
                    assert budget == 0
                    nodes.pop()
                    if len(nodes) > 2:
                        nodes.pop()
                        valid -= 1

                assert RoadPolygon.from_nodes(nodes).is_valid()
                assert 0 <= valid <= self.num_control_nodes

        return nodes

    def generate(self):
        """Generates a random road and returns the control nodes, the sample nodes, and the generation boundary.
        Note that because of the Catmull-Rom model the control nodes will consist of two extra nodes, one at the start,
        and one at the end."""
        control_nodes = self.generate_control_nodes()
        sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)
        while (not RoadPolygon.from_nodes(sample_nodes).is_valid() and
               self.generation_boundary.contains(RoadPolygon.from_nodes(control_nodes[1:-1]))):
            control_nodes = self.generate_control_nodes()
            sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)
        return control_nodes, sample_nodes, self.generation_boundary

    def _get_initial_point(self) -> Point:
        """Gets the 2D point representing the initial road node."""
        return Point(self.initial_node[0], self.initial_node[1])

    def _get_initial_control_node(self) -> Point4D:
        """Gets the 4D point representing the first valid road control node."""
        x0, y0, z, width = self.initial_node
        x, y = self._get_next_xy(x0, y0, 270)
        assert not(self.generation_boundary.bbox.contains(Point(x, y)))

        return x, y, z, width

    def _get_next_node(self, first_node: Point4D, second_node: Point4D, max_angle: int) -> Point4D:
        """Gets the next node constrained by 'max_angle'."""
        v = np.subtract(second_node, first_node)
        start_angle = int(np.degrees(np.arctan2(v[1], v[0])))
        angle = randint(start_angle - max_angle, start_angle + max_angle)
        x0, y0, z0, width0 = second_node
        x1, y1 = self._get_next_xy(x0, y0, angle)
        return x1, y1, z0, width0

    def _get_next_xy(self, x0: float, y0: float, angle: float) -> Point2D:
        """Gets the next 2D point constrained by 'angle'."""
        angle_rad = math.radians(angle)
        return x0 + self.seg_length * math.cos(angle_rad), y0 + self.seg_length * math.sin(angle_rad)

    def _get_next_max_angle(self, i: int, threshold=NUM_INITIAL_SEGMENTS_THRESHOLD) -> int:
        """Gets the next angle to constrain node generation."""
        if i < threshold or i == self.num_control_nodes - 1:
            return 0
        else:
            return self.max_angle


if __name__ == "__main__":
    cnt_nodes, smp_nodes, _ = RoadGenerator(num_control_nodes=10, num_spline_nodes=20).generate()

    from self_driving.curve_interpolation import plot_catmull_rom

    c = [(n[0], n[1]) for n in smp_nodes]
    ps = [(n[0], n[1]) for n in cnt_nodes]
    plot_catmull_rom(c, ps)

    rp = RoadPolygon.from_nodes([(0, 0, -28, 8),
                                 (0, 4, -28, 8),
                                 (5, 15, -28, 8),
                                 (20, -4, -28, 8)])

    assert not rp.is_valid(), "It should be invalid"

    rp = RoadPolygon.from_nodes([(0, 0, -28, 8),
                                 (3, 2, -28, 8),
                                 (10, -1, -28, 8)])

    assert rp.is_valid(), "It should be valid"
