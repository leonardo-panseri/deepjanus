import os.path
from datetime import datetime

from deepjanus import nsga2
from deepjanus.log import log_setup
from deepjanus.seed_pool import SeedFileGenerator
from deepjanus_mnist.mnist_config import MNISTConfig
from deepjanus_mnist.mnist_problem import MNISTProblem


def execute_deepjanus(problem: MNISTProblem, restart_from_last_gen):
    nsga2.main(problem, restart_from_last_gen=restart_from_last_gen)


def generate_seeds(problem: MNISTProblem, folder_name='generated', quantity=100):
    from deepjanus_mnist.seed_generator import seed_candidate_generator
    seed_generator = SeedFileGenerator([problem], folder_name, seed_candidate_generator(5))
    seed_generator.generate_seeds(quantity)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='DeepJanus-MNIST', description='Evolutionary algorithm that searches the input '
                                                                         'space of a digit recognition system to identify '
                                                                         'its frontier of behaviors')
    parser.add_argument('-s', '--seeds', help='generate seeds from MNIST dataset', action='store_true')
    parser.add_argument('-c', '--config', type=str, help='load config from file')
    parser.add_argument('-r', dest='restart_from_last_gen', action='store_true',
                        help='restarts experiment from last generation')

    subparsers = parser.add_subparsers(dest='subcmd')

    parser_train = subparsers.add_parser('train', description='Trains a CNN model for handwritten digit '
                                                              'classification on the MNIST dataset')
    parser_train.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=12)
    parser_train.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)

    args = parser.parse_args()

    proj_root = os.path.dirname(__file__)
    if args.config:
        cfg = MNISTConfig.from_file(args.config, proj_root)
    else:
        cfg = MNISTConfig(proj_root)
    prob = MNISTProblem(cfg)

    if args.subcmd == 'train':
        from deepjanus_mnist.model_trainer import train_model
        train_model(str(cfg.FOLDERS.models.joinpath('cnnClassifier_trained')),
                    args.batch_size, args.nb_epoch)
    elif args.seeds:
        generate_seeds(prob)
    else:
        # Disable TensorFlow logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        execute_deepjanus(prob, args.restart_from_last_gen)
