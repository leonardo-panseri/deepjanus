import json
import os
import signal
import sys
from datetime import datetime

from matplotlib import pyplot as plt

from deepjanus import nsga2
from deepjanus.archive import SmartArchive
from deepjanus.folders import SeedStorage
from deepjanus.log import get_logger, log_setup
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_problem import BeamNGProblem

log = get_logger(__file__)


def execute_deepjanus(problem: BeamNGProblem):
    nsga2.main(problem)

    # Needed?
    plt.ioff()
    plt.show()


def generate_seeds(problem1: BeamNGProblem, problem2: BeamNGProblem | None, folder_name='generated_seeds', quantity=12):
    good_members_found = 0
    attempts = 0
    storage = SeedStorage(problem1.config, folder_name)

    def is_outside_frontier(member: BeamNGMember, problem: BeamNGProblem):
        member.clear_evaluation()
        return not member.evaluate(problem.get_evaluator())

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
                                                                       'its frontier of behaviors')
    parser.add_argument('-s', '--seeds', action='store_true', help='generate seeds')
    subparsers = parser.add_subparsers(dest='subcmd')

    parser_train = subparsers.add_parser('train', description='Lane-keeping assist system behavioral cloning '
                                                              'training program')
    parser_train.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser_train.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser_train.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=200)
    parser_train.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)
    parser_train.add_argument('-a', help='save all, not best models only', dest='save_best_only',
                              action='store_false', default=True)
    parser_train.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)

    parser_gen_training = subparsers.add_parser('generate-training', description='Lane-keeping assist system '
                                                                                 'training data generator')
    parser_gen_training.add_argument('-i', help='number of roads to generate', dest='iterations',
                                     type=int, default=12)

    args = parser.parse_args()

    cfg = BeamNGConfig(os.path.dirname(__file__))
    prob = BeamNGProblem(cfg, SmartArchive(cfg.ARCHIVE_THRESHOLD))
    log_setup.use_ini(cfg.FOLDERS.log_ini)
    log_setup.setup_log_file(prob.experiment_path
                             .joinpath(datetime.strftime(datetime.now(), '%d-%m-%Y_%H-%M-%S') + '.log'))

    def signal_handler(_, __):
        print('Run interrupted by user')
        if prob.get_evaluator().bng:
            prob.get_evaluator().bng.beamng_close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if args.subcmd == 'train':
        from training import train_from_recordings
        train_from_recordings.main(args)
    elif args.subcmd == 'generate-training':
        from training import train_dataset_recorder
        train_dataset_recorder.main(args.iterations)
    elif args.seeds:
        cfg_lq = BeamNGConfig(os.path.dirname(__file__))
        cfg_lq.BEAMNG_PORT += 1
        cfg_lq.MODEL_FILE = 'self-driving-car-4600'
        prob_lq = BeamNGProblem(cfg_lq, SmartArchive(cfg.ARCHIVE_THRESHOLD))
        generate_seeds(prob, prob_lq)
    else:
        # Disable TensorFlow logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        import tensorflow.python.util.module_wrapper as mw
        mw._PER_MODULE_WARNING_LIMIT = 0

        execute_deepjanus(prob)
