import json
import signal
import sys

from matplotlib import pyplot as plt

from core import nsga2
from core.archive import SmartArchive
from core.folders import FOLDERS, SeedStorage
from core.log import get_logger, configure_logging
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_problem import BeamNGProblem

log = get_logger(__file__)


def execute_deepjanus(problem: BeamNGProblem):
    def signal_handler(sig, frame):
        print('Run interrupted by user')
        problem.get_evaluator().bng.beamng_close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    nsga2.main(problem)

    # Needed?
    plt.ioff()
    plt.show()


def generate_seeds(problem1: BeamNGProblem, problem2: BeamNGProblem | None, folder_name='generated_seeds', quantity=12):
    good_members_found = 0
    attempts = 0
    storage = SeedStorage(folder_name)

    def is_outside_frontier(member: BeamNGMember, problem: BeamNGProblem):
        member.clear_evaluation()
        member.evaluate(problem.get_evaluator())
        if member.distance_to_frontier is None or member.distance_to_frontier <= 0:
            return True
        return False

    while good_members_found < quantity:
        seed_index = good_members_found + 1
        path = storage.get_path_by_index(seed_index)
        if path.exists():
            log.info(f'Seed{seed_index} already generated')
            good_members_found += 1
            continue
        attempts += 1
        log.info(f'Total attempts: {attempts}; Found {good_members_found}/{quantity}; Looking for seed{seed_index}')

        mbr = problem1.generate_random_member()
        if is_outside_frontier(mbr, problem1):
            continue

        if problem2:
            mbr = problem2.member_class().from_dict(mbr.to_dict())
            if is_outside_frontier(mbr, problem2):
                continue

        mbr.clear_evaluation()
        path.write_text(json.dumps(mbr.to_dict()))

        good_members_found += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='DeepJanus-BNG', description='Evolutionary algorithm that searches the input '
                                                                       'space of a lane-keep assist system to identify '
                                                                       'its frontier of behaviors.')
    parser.add_argument('-s', '--seeds', action='store_true', help='Generate seeds.')
    args = parser.parse_args()

    cfg = BeamNGConfig()
    prob = BeamNGProblem(cfg, SmartArchive(cfg.ARCHIVE_THRESHOLD))
    configure_logging(FOLDERS.log_ini)

    if args.seeds:
        cfg_lq = BeamNGConfig()
        cfg_lq.BEAMNG_PORT += 1
        cfg_lq.MODEL_FILE = 'self-driving-car-4600'
        prob_lq = BeamNGProblem(cfg_lq, SmartArchive(cfg.ARCHIVE_THRESHOLD))
        generate_seeds(prob, prob_lq)
    else:
        execute_deepjanus(prob)
