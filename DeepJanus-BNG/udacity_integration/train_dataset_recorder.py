import numpy as np

from core.log import get_logger
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_interface import BeamNGInterface
from self_driving.beamng_roads import BeamNGRoad
from self_driving.shapely_roads import RoadGenerator
from udacity_integration.training_data_collector import TrainingDataCollector
from self_driving.points import Point4D, Point2D, points_distance

log = get_logger(__file__)


def calculate_script(road_points: list[Point4D]) -> tuple[list[dict[str, float]], list[Point2D]]:
    """Calculates the script that the BeamNG AI should follow. It contains a sequence of points representing
    the middle of the lane where the AI should drive."""
    def calculate_lane_center(p1, p2) -> Point2D:
        a = np.subtract(p2[0:2], p1[0:2])
        # Calculate the vector which length is half the road width
        road_dir = (a / np.linalg.norm(a)) * p1[3] / 4
        # Add normal vector
        r = p1[0:2] + np.array([road_dir[1], -road_dir[0]])
        return tuple(r)

    script_points = [calculate_lane_center(road_points[i], road_points[i + 1]) for i in range(len(road_points) - 1)]
    assert(len(script_points) == len(road_points)-1)
    # Get the last script point
    script_points += [calculate_lane_center(road_points[-1], road_points[-2])]
    assert (len(script_points) == len(road_points))
    orig = script_points[0]

    script = [{'x': orig[0], 'y': orig[1], 'z': .5, 't': .0}]
    i = 1
    time = 0.18
    goal = len(script_points) - 1

    while i < goal:
        node = {
            'x': script_points[i][0],
            'y': script_points[i][1],
            'z': .5,
            't': time,
        }
        script.append(node)
        i += 1
        time += 0.18
    return script, script_points


def run_sim(config: BeamNGConfig, road: BeamNGRoad):
    """Runs a simulation on the given road."""
    script, script_points = calculate_script(road.nodes)

    bng = BeamNGInterface(config)

    bng.setup_road(road)
    bng.setup_vehicle(True)

    sim_data_collector = TrainingDataCollector(bng.vehicle, road, script_points)

    bng.beamng_bring_up()

    # Trick: we start from the road center
    bng.vehicle.vehicle.ai.set_script(script[4:])

    bng.beamng_step(1)

    try:
        for idx in range(1000):
            last_state = sim_data_collector.collect_and_write_current_data()
            dist = points_distance(last_state.pos, road.waypoint_goal.position)
            if dist < 15.0:
                break

            bng.beamng_step()
    finally:
        bng.beamng_close()


def main(iterations):
    for _ in range(iterations):
        log.info('Generating road...')
        cnt_nodes, smp_nodes, _ = RoadGenerator(num_control_nodes=20).generate()
        rd = BeamNGRoad(smp_nodes, cnt_nodes)

        log.info('Running simulation...')
        run_sim(BeamNGConfig(), rd)


if __name__ == '__main__':
    main(12)
