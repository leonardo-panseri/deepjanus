import json
from datetime import datetime
import os.path

import h5py
import numpy as np
from deepjanus import nsga2
from deepjanus.archive import SmartArchive
from deepjanus.folders import SeedStorage
from deepjanus.log import log_setup

from deepjanus_mnist.mnist_config import MNISTConfig
from deepjanus_mnist.mnist_member import MNISTMember
from deepjanus_mnist.mnist_problem import MNISTProblem
from deepjanus_mnist.image_tools import bitmap_to_svg


def execute_deepjanus(problem: MNISTProblem):
    nsga2.main(problem)


def prepare_seeds(h5_file: str, config: MNISTConfig, output_folder: str):
    storage = SeedStorage(config, output_folder)
    dataset = h5py.File(h5_file, 'r')
    x = np.array(dataset.get('xn'))
    y = np.array(dataset.get('yn'))
    assert len(x) == len(y)
    for i in range(len(x)):
        bitmap = x[i]
        svg = bitmap_to_svg(bitmap)
        label = int(y[i])
        seed = MNISTMember(svg, label, f'seed{i}')

        path = storage.get_path_by_index(i)
        path.write_text(json.dumps(seed.to_dict(), indent=2))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='DeepJanus-MNIST', description='Evolutionary algorithm that searches the input '
                                                                         'space of a digit recognition system to identify '
                                                                         'its frontier of behaviors')
    parser.add_argument('-s', '--seeds', help='generate seeds from a MNIST subset', dest='seed_dataset', type=str)

    cfg = MNISTConfig(os.path.dirname(__file__))
    prob = MNISTProblem(cfg, SmartArchive(cfg.ARCHIVE_THRESHOLD))
    log_setup.use_ini(cfg.FOLDERS.log_ini)
    log_setup.setup_log_file(prob.experiment_path
                             .joinpath(datetime.strftime(datetime.now(), '%d-%m-%Y_%H-%M-%S') + '.log'))

    args = parser.parse_args()

    if args.seed_dataset:
        prepare_seeds(args.seed_dataset, cfg, 'generated')
    else:
        execute_deepjanus(prob)
