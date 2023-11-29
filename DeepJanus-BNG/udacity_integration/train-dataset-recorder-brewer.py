import random
from time import sleep
from typing import Tuple, List

import numpy as np

from self_driving.beamng_interface import BeamNGInterface
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_vehicles import BeamNGVehicleCameras
from self_driving.beamng_roads import BeamNGWaypoint, DecalRoad, BeamNGRoad
from self_driving.shapely_roads import RoadGenerator
from udacity_integration.training_data_collector_and_writer import TrainingDataCollectorAndWriter
from self_driving.points import to_3d_point

# maps.install_map_if_needed()
STEPS = 5


# x is -y and *angle direction is reversed*
def get_rotation(road: BeamNGRoad):
    v1 = road.nodes[0][:2]
    v2 = road.nodes[1][:2]
    v = np.subtract(v1, v2)
    deg = np.degrees(np.arctan2([v[0]], [v[1]]))
    return 0, 0, deg


def get_script_point(p1, p2) -> Tuple[Tuple, Tuple]:
    a = np.subtract(p2[0:2], p1[0:2])

    # calculate the vector which length is half the road width
    v = (a / np.linalg.norm(a)) * p1[3] / 4

    # add normal vectors
    r = p1[0:2] + np.array([v[1], -v[0]])
    return tuple(r)

# Calculate the points to guide the AI from the road points
def calculate_script(road_points):
    script_points = [get_script_point(road_points[i], road_points[i + 1]) for i in range(len(road_points) - 1)]
    assert(len(script_points) == len(road_points)-1)
    # Get the last script point
    script_points += [get_script_point(road_points[-1], road_points[-2])]
    assert (len(script_points) == len(road_points))
    orig = script_points[0]

    script = [{'x': orig[0], 'y': orig[1], 'z': .5, 't': 0}]
    i = 1
    time = 0.18
    # goal = len(street_1.nodes) - 1
    # goal = len(brewer.road_points.right) - 1
    goal = len(script_points) - 1

    while i < goal:
        node = {
            # 'x': street_1.nodes[i][0],
            # 'y': street_1.nodes[i][1],
            # 'x': brewer.road_points.right[i][0],
            # 'y': brewer.road_points.right[i][1],
            'x': script_points[i][0],
            'y': script_points[i][1],
            'z': .5,
            't': time,
        }
        script.append(node)
        i += 1
        time += 0.18
    return script


def distance(p1, p2):
    return np.linalg.norm(np.subtract(to_3d_point(p1), to_3d_point(p2)))


def run_sim(street_1: BeamNGRoad):
    # TODO fix simulation code
    brewer = BeamNGInterface(BeamNGConfig(), road_nodes=street_1.nodes)
    waypoint_goal = BeamNGWaypoint('waypoint_goal', to_3d_point(street_1.nodes[-1]))

    vehicle = brewer.setup_vehicle()
    camera = brewer.setup_scenario_camera()
    beamng = brewer._bng
    brewer.setup_road(street_1.nodes)

    maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

    cameras = BeamNGVehicleCameras()

    brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()
    #brewer.vehicle_start_pose = BeamNGPose()

    sim_data_collector = TrainingDataCollectorAndWriter(vehicle, beamng, street_1, cameras)

    brewer.beamng_bring_up()
    print('bring up ok')

    script = calculate_script(brewer.road_points.middle)

    # Trick: we start from the road center
    vehicle.ai_set_script(script[4:])

    #vehicle.ai_drive_in_lane(True)
    beamng.pause()
    beamng.step(1)

    def start():
        for idx in range(1000):
            if (idx * 0.05 * STEPS) > 3.:
                sim_data_collector.collect_and_write_current_data()
                dist = distance(sim_data_collector.last_state.pos, waypoint_goal.position)
                if dist < 15.0:
                    beamng.resume()
                    break

            # one step is 0.05 seconds (5/100)
            beamng.step(STEPS)

    try:
        start()
    finally:

        beamng.close()


if __name__ == '__main__':
    cnt_nodes, smp_nodes, _ = RoadGenerator(num_control_nodes=20, max_angle=80, seg_length=25,
                                            num_spline_nodes=20).generate()

    # plot_road(road, save=True)

    street = BeamNGRoad(smp_nodes, material='')

    run_sim(street)
