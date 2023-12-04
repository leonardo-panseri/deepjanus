from typing import Literal

from self_driving.beamng_interface import BeamNGInterface
from self_driving.oob_monitor import OutOfBoundsMonitor
from self_driving.shapely_roads import RoadPolygon
from self_driving.simulation_data import SimulationDataRecords, SimulationData, SimulationDataRecord
from self_driving.points import points_distance


class SimulationDataCollector:
    """Utility to collect all relevant data about a simulation"""

    def __init__(self, bng: BeamNGInterface, simulation_name: str = None):
        self.bng = bng
        self.name = simulation_name

        self.oob_monitor = OutOfBoundsMonitor(RoadPolygon(self.bng.road), self.bng.vehicle)

        self.states: SimulationDataRecords = []
        self.simulation_data: SimulationData = SimulationData(simulation_name)
        self.simulation_data.set(self.bng.params, self.bng.road, self.states)
        self.simulation_data.clean()

    def collect_current_data(self, oob_bb=True, lane: Literal['right', 'left'] = 'right'):
        """Collects the data about the current state of the simulation.
        If oob_bb is True, then the out-of-bound monitor will use the bounding box of the car."""
        self.bng.vehicle.update_state()
        car_state = self.bng.vehicle.get_state()

        is_oob, oob_counter, max_oob_percentage, oob_distance = self.oob_monitor.is_out_of_bounds(oob_bb=oob_bb,
                                                                                                  lane=lane)

        dist_from_goal = points_distance(car_state.pos, self.bng.road.waypoint_goal.position)

        sim_data_record = SimulationDataRecord(**car_state._asdict(),
                                               is_oob=is_oob,
                                               oob_counter=oob_counter,
                                               max_oob_percentage=max_oob_percentage,
                                               oob_distance=oob_distance,
                                               dist_from_goal=dist_from_goal)
        self.states.append(sim_data_record)
        return sim_data_record

    def get_simulation_data(self) -> SimulationData:
        """Gets the simulation data."""
        return self.simulation_data

    def take_car_picture_if_needed(self):
        """Takes a picture of the car and saves it on disk if the vehicle is out of bounds."""
        if not self.states:
            return
        last_state = self.states[-1]
        if last_state.is_oob:
            img_path = self.simulation_data.path_root.joinpath(f'oob_camera_shot{last_state.oob_counter}.jpg')
            img_path.parent.mkdir(parents=True, exist_ok=True)
            if not img_path.exists():
                vehicle_pos = last_state.pos
                cam_pos = (vehicle_pos[0], vehicle_pos[1] + 5.0, vehicle_pos[2] + 5.0)
                cam_dir = (vehicle_pos[0] - cam_pos[0], vehicle_pos[1] - cam_pos[1], vehicle_pos[2] - cam_pos[2])
                self.bng.capture_image(cam_pos, cam_dir).save(str(img_path))

    def save(self):
        """Saves the simulation data to disk."""
        self.simulation_data.save()
