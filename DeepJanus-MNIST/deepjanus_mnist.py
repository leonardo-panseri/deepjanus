import os.path
from datetime import datetime

from deepjanus import nsga2
from deepjanus.log import log_setup
from deepjanus.seed_pool import SeedFileGenerator
from deepjanus_mnist.mnist_config import MNISTConfig
from deepjanus_mnist.mnist_problem import MNISTProblem


def execute_deepjanus(problem: MNISTProblem):
    nsga2.main(problem)


def generate_seeds(problem1: MNISTProblem, problem2: MNISTProblem, folder_name='generated', quantity=100):
    from deepjanus_mnist.seed_generator import seed_candidate_generator
    seed_generator = SeedFileGenerator([problem1, problem2], folder_name, seed_candidate_generator(5))
    seed_generator.generate_seeds(quantity)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='DeepJanus-MNIST', description='Evolutionary algorithm that searches the input '
                                                                         'space of a digit recognition system to identify '
                                                                         'its frontier of behaviors')
    parser.add_argument('-s', '--seeds', help='generate seeds from MNIST dataset', action='store_true')

    subparsers = parser.add_subparsers(dest='subcmd')

    parser_train = subparsers.add_parser('train', description='Trains a CNN model for handwritten digit '
                                                              'classification on the MNIST dataset')
    parser_train.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=12)
    parser_train.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)

    cfg = MNISTConfig(os.path.dirname(__file__))
    prob = MNISTProblem(cfg)
    log_setup.setup_console_log(cfg.FOLDERS.log_ini)
    log_setup.setup_file_log(prob.experiment_path
                             .joinpath(datetime.strftime(datetime.now(), '%d-%m-%Y_%H-%M-%S') + '.log'))

    args = parser.parse_args()

    if args.subcmd == 'train':
        from deepjanus_mnist.model_trainer import train_model
        train_model(str(cfg.FOLDERS.models.joinpath('cnnClassifier_trained')),
                    args.batch_size, args.nb_epoch)
    elif args.seeds:
        cfg_lq = MNISTConfig(os.path.dirname(__file__))
        cfg_lq.MODEL_FILE = 'cnnClassifier_lowLR'
        prob_lq = MNISTProblem(cfg_lq)
        generate_seeds(prob, prob_lq)
    else:
        # Disable TensorFlow logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        execute_deepjanus(prob)
