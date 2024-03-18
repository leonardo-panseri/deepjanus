import os
from datetime import datetime

from deepjanus import nsga2
from deepjanus.log import get_logger, log_setup
from deepjanus.seed_pool import SeedFileGenerator
from deepjanus_bng.beamng_config import BeamNGConfig
from deepjanus_bng.beamng_problem import BeamNGProblem

import matplotlib
matplotlib.use('Agg')

log = get_logger(__file__)


def execute_deepjanus(problem: BeamNGProblem, restart_from_last_gen):
    nsga2.main(problem, restart_from_last_gen=restart_from_last_gen)


def generate_seeds(problem1: BeamNGProblem, folder_name='generated', quantity=12):
    def seed_candidate_generator():
        while True:
            yield problem1.generate_random_member()

    seed_generator = SeedFileGenerator([problem1], folder_name, seed_candidate_generator())
    seed_generator.generate_seeds(quantity)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='DeepJanus-BNG', description='Evolutionary algorithm that searches the input '
                                                                       'space of a lane-keep assist system to identify '
                                                                       'its frontier of behaviors')
    parser.add_argument('-s', '--seeds', action='store_true', help='generate seeds')
    parser.add_argument('-c', '--config', type=str, help='load config from file')
    parser.add_argument('-r', dest='restart_from_last_gen', action='store_true',
                        help='restarts experiment from last generation')
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

    proj_root = os.path.dirname(__file__)
    if args.config:
        cfg = BeamNGConfig.from_file(args.config, proj_root)
    else:
        cfg = BeamNGConfig(proj_root)
    prob = BeamNGProblem(cfg)

    if args.subcmd == 'train':
        from deepjanus_bng.training import train_from_recordings
        train_from_recordings.main(args)
    elif args.subcmd == 'generate-training':
        from deepjanus_bng.training import train_dataset_recorder
        train_dataset_recorder.main(args.iterations)
    elif args.seeds:
        prob.config.PARALLEL_EVALS = 0
        generate_seeds(prob)
    else:
        # Disable TensorFlow logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        import tensorflow.python.util.module_wrapper as mw
        mw._PER_MODULE_WARNING_LIMIT = 0

        execute_deepjanus(prob, args.restart_from_last_gen)
