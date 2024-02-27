import os
from datetime import datetime

from deepjanus import nsga2
from deepjanus.log import get_logger, log_setup
from deepjanus.seed_pool import SeedFileGenerator
from deepjanus_bng.beamng_config import BeamNGConfig
from deepjanus_bng.beamng_problem import BeamNGProblem

log = get_logger(__file__)


def execute_deepjanus(problem: BeamNGProblem):
    # Save BeamNGpy logs to file
    import logging
    from pathlib import Path
    l = logging.getLogger('beamngpy')
    userpath = Path(problem.config.BEAMNG_USER_DIR).parent
    userpath.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(userpath.joinpath('sim.log'), 'w')
    fh.setFormatter(logging.Formatter(r'[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s', '%H:%M:%S'))
    l.addHandler(fh)
    l.setLevel(logging.DEBUG)

    nsga2.main(problem)


def generate_seeds(problem1: BeamNGProblem, problem2: BeamNGProblem, folder_name='generated', quantity=12):
    def seed_candidate_generator():
        while True:
            yield problem1.generate_random_member()

    seed_generator = SeedFileGenerator([problem1, problem2], folder_name, seed_candidate_generator())
    seed_generator.generate_seeds(quantity)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='DeepJanus-BNG', description='Evolutionary algorithm that searches the input '
                                                                       'space of a lane-keep assist system to identify '
                                                                       'its frontier of behaviors')
    parser.add_argument('-s', '--seeds', action='store_true', help='generate seeds')
    parser.add_argument('-c', '--config', type=str, help='load config from file')
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
    log_setup.setup_console_log(cfg.FOLDERS.log_ini)
    log_setup.setup_file_log(prob.experiment_path
                             .joinpath(datetime.strftime(datetime.now(), '%d-%m-%Y_%H-%M-%S') + '.log'))

    if args.subcmd == 'train':
        from deepjanus_bng.training import train_from_recordings
        train_from_recordings.main(args)
    elif args.subcmd == 'generate-training':
        from deepjanus_bng.training import train_dataset_recorder
        train_dataset_recorder.main(args.iterations)
    elif args.seeds:
        cfg_lq = BeamNGConfig(os.path.dirname(__file__))
        cfg_lq.BEAMNG_PORT += 1
        cfg_lq.MODEL_FILE = 'self-driving-car-4600'
        prob_lq = BeamNGProblem(cfg_lq)
        generate_seeds(prob, prob_lq)
    else:
        # Disable TensorFlow logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        import tensorflow.python.util.module_wrapper as mw
        mw._PER_MODULE_WARNING_LIMIT = 0

        execute_deepjanus(prob)
