import os.path
from datetime import datetime

from deepjanus import nsga2
from deepjanus.archive import SmartArchive
from deepjanus.log import log_setup
from deepjanus.seed_pool import SeedFileGenerator
from deepjanus_mnist.mnist_config import MNISTConfig
from deepjanus_mnist.mnist_problem import MNISTProblem
from deepjanus_mnist.seed_generator import seed_candidate_generator


def execute_deepjanus(problem: MNISTProblem):
    nsga2.main(problem)


def generate_seeds(problem1: MNISTProblem, problem2: MNISTProblem, folder_name='generated', quantity=100):
    seed_generator = SeedFileGenerator([problem1, problem2], folder_name, seed_candidate_generator(5))
    seed_generator.generate_seeds(quantity)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='DeepJanus-MNIST', description='Evolutionary algorithm that searches the input '
                                                                         'space of a digit recognition system to identify '
                                                                         'its frontier of behaviors')
    parser.add_argument('-s', '--seeds', help='generate seeds from MNIST dataset', action='store_true')

    cfg = MNISTConfig(os.path.dirname(__file__))
    prob = MNISTProblem(cfg, SmartArchive(cfg.TARGET_ERROR, cfg.ARCHIVE_THRESHOLD))
    log_setup.setup_console_log(cfg.FOLDERS.log_ini)
    log_setup.setup_file_log(prob.experiment_path
                             .joinpath(datetime.strftime(datetime.now(), '%d-%m-%Y_%H-%M-%S') + '.log'))

    args = parser.parse_args()

    if args.seeds:
        cfg_lq = MNISTConfig(os.path.dirname(__file__))
        cfg_lq.MODEL_FILE = 'cnnClassifier_lowLR'
        prob_lq = MNISTProblem(cfg_lq, SmartArchive(cfg.TARGET_ERROR, cfg.ARCHIVE_THRESHOLD))
        generate_seeds(prob, prob_lq)
    else:
        execute_deepjanus(prob)
