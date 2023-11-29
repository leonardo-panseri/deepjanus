import os
from collections import namedtuple
from datetime import datetime

from matplotlib import pyplot as plt

from core.folders import FOLDERS
from self_driving.beamng_roads import BeamNGRoad
from self_driving.points import Point2D
from self_driving.shapely_roads import RoadPolygon
from self_driving.oob_monitor import OutOfBoundsMonitor
from self_driving.beamng_vehicles import BeamNGVehicle

CSV_HEADER = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
CSVEntry = namedtuple('CSVEntry', CSV_HEADER)


class TrainingDataCollector:
    """Collects simulation data for training a ML model."""

    def __init__(self, vehicle: BeamNGVehicle, road: BeamNGRoad, script_points: list[Point2D]):
        self.vehicle = vehicle
        self.road = road
        self.script_points = script_points

        self.log_folder = FOLDERS.training_recordings.joinpath(datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
        self.oob_monitor = OutOfBoundsMonitor(RoadPolygon(road), self.vehicle)
        self.sequence_index = 0

        os.makedirs(self.log_folder, exist_ok=True)
        self.append_line(CSV_HEADER)
        self.save_road_image()

    def save_road_image(self):
        """Saves an image with a representation of the simulated road and the path that the car will follow."""
        fig, ax = plt.subplots()
        self.road.to_image(ax)
        tup = list(zip(*self.script_points))
        ax.plot(tup[0], tup[1], color='red', linewidth=0.5)
        fig.savefig(self.log_folder.joinpath('road_with_script.svg'))

    def collect_and_write_current_data(self):
        """Collects current simulation data and saves them to disk."""
        self.sequence_index += 1
        self.vehicle.update_state()
        car_state = self.vehicle.get_state()

        img_names = []
        for cam in ['center', 'left', 'right']:
            img = self.vehicle.capture_image(cam)
            img_name = 'z{:05d}_{}_{}.jpg'.format(self.sequence_index, car_state.steering_input, cam)
            img.save(os.path.join(self.log_folder, img_name))
            img_names.append(img_name)

        values = CSVEntry(*img_names, car_state.steering_input, car_state.throttle, car_state.brake, car_state.vel_kmh)
        self.append_line(list(values))

        return car_state

    def append_line(self, values: list[str]):
        """Appends a line to the CSV file containing the simulation log."""
        with open(os.path.join(self.log_folder, 'driving_log.csv'), 'a+') as f:
            f.write(','.join(str(v) for v in values) + '\n')
