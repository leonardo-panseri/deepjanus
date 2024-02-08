from typing import Literal

from shapely.geometry import Point, Polygon

from .shapely_roads import RoadPolygon
from .beamng_vehicles import BeamNGVehicle


class OutOfBoundsMonitor:
    """Utility to check if a vehicle has exited its lane bounds"""

    def __init__(self, road_polygon: RoadPolygon, vehicle: BeamNGVehicle):
        self.road_polygon = road_polygon
        self.vehicle = vehicle
        self.oob_counter = 0
        self.last_is_oob = False
        self.last_max_oob_percentage = .0

    def is_out_of_bounds(self, lane: Literal['right', 'left'] = 'right', oob_bb=True,
                         tolerance=0.05) -> tuple[bool, int, float, float]:
        """Checks if the vehicle has exited the lane bounds. Returns a tuple containing the result, the out-of-bounds
        count, the maximum out-of-bounds percentage for the last oob event (NaN if oob_bb=False), and the distance
        from the bound (NaN if oob_bb=True)."""
        max_oob_percentage = float("nan")
        oob_distance = float("nan")

        if oob_bb:
            is_oob = self.is_oob_bb(tolerance=tolerance, lane=lane)
            self.update_oob_percentage(is_oob)
            max_oob_percentage = self.last_max_oob_percentage
        else:
            is_oob = self.is_oob(lane=lane)
            oob_distance = self.oob_distance(lane=lane)

        self.update_oob_counter(is_oob)

        return is_oob, self.oob_counter, max_oob_percentage, oob_distance

    def update_oob_counter(self, is_oob: bool):
        """Updates the out-of-bound events counter."""
        # Check last_is_obb to prevent counting the same event multiple times
        if not self.last_is_oob and is_oob:
            self.oob_counter += 1
            self.last_is_oob = True
        elif self.last_is_oob and not is_oob:
            self.last_is_oob = False

    def update_oob_percentage(self, is_oob: bool):
        """Updates the out-of-bounds max percentage for the last event."""
        if not self.last_is_oob and is_oob:
            self.last_max_oob_percentage = self.oob_percentage()
        elif self.last_is_oob and is_oob:
            self.last_max_oob_percentage = max(self.last_max_oob_percentage, self.oob_percentage())

    def oob_percentage(self, lane: Literal['right', 'left'] = 'right') -> float:
        """Returns the percentage of the bounding box of the car with respect to
        one of the lanes of the road or the road itself (depending on the value of wrt)."""
        car_bbox_polygon = self._get_car_bbox_polygon()
        if lane == 'right':
            intersection = car_bbox_polygon.intersection(self.road_polygon.right_polygon)
        elif lane == 'left':
            intersection = car_bbox_polygon.intersection(self.road_polygon.left_polygon)
        else:
            intersection = car_bbox_polygon.intersection(self.road_polygon.polygon)
        return 1 - intersection.area / car_bbox_polygon.area

    def is_oob_bb(self, tolerance=0.05, lane: Literal['right', 'left'] = 'right') -> bool:
        """Returns true if the bounding box of the car is more than tolerance percentage outside the road."""
        return self.oob_percentage(lane=lane) > tolerance

    def oob_distance(self,  lane: Literal['right', 'left'] = 'right') -> float:
        """Returns the difference between the width of a lane and the distance between the car and the center
        of the road."""
        car_point = Point(self.vehicle.get_state().pos)
        divisor = 4.0
        if lane == 'right':
            distance = self.road_polygon.right_polyline.distance(car_point)
        elif lane == 'left':
            distance = self.road_polygon.left_polyline.distance(car_point)
        else:
            distance = self.road_polygon.polyline.distance(car_point)
            divisor = 2.0
        difference = self.road_polygon.road_width / divisor - distance
        return difference

    def is_oob(self, lane: Literal['right', 'left'] = 'right') -> bool:
        """Returns true if the car is an out-of-bound (OOB).

        The OOB can be calculated with respect to the left or right lanes,
        or with respect to the whole road.

        The car position is represented by the center of mass of the car.
        If you want to calculate the OOBs using the bounding box of the car,
        call self.is_oob_bb."""
        car_point = Point(self.vehicle.get_state().pos)
        if lane == 'right':
            return not self.road_polygon.right_polygon.contains(car_point)
        elif lane == 'left':
            return not self.road_polygon.left_polygon.contains(car_point)
        else:
            return not self.road_polygon.polygon.contains(car_point)

    def _get_car_bbox_polygon(self) -> Polygon:
        """Gets the polygon representing the car bounding box."""
        car_bbox = self.vehicle.get_vehicle_bbox()

        # x coordinates of the bounding box of the car.
        boundary_x = [
            car_bbox['near_bottom_left'][0],
            car_bbox['near_bottom_right'][0],
            car_bbox['far_bottom_right'][0],
            car_bbox['far_bottom_left'][0],
            car_bbox['near_bottom_left'][0],
        ]

        # y coordinates of the bounding box of the car.
        boundary_y = [
            car_bbox['near_bottom_left'][1],
            car_bbox['near_bottom_right'][1],
            car_bbox['far_bottom_right'][1],
            car_bbox['far_bottom_left'][1],
            car_bbox['near_bottom_left'][1],
        ]

        return Polygon(zip(boundary_x, boundary_y))
