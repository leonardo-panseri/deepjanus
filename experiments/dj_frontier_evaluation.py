import json
from pathlib import Path
import glob

import numpy as np

from deepjanus_mnist.mnist_individual import MNISTIndividual
from deepjanus_mnist.mnist_member import MNISTMember
from deepjanus_mnist.mnist_problem import MNISTProblem
from deepjanus_mnist.mnist_config import MNISTConfig
from deepjanus_mnist.image_tools import bitmap_to_svg


class FakeFitness:
    values = None


if __name__ == '__main__':
    config = MNISTConfig('../../DeepJanus-MNIST')
    problem = MNISTProblem(config)

    mnist_experiments_path = Path('./experiment-MNIST-FSE/HQ')
    results = {}
    for i in range(1, 11):
        results[i] = []
        archive_path = mnist_experiments_path.joinpath(str(i), 'results', 'archive')
        member_files = glob.glob('*.npy', root_dir=archive_path)
        num_files = len(member_files)
        print(f'### Evaluating results of experiment {i}')
        for j in range(0, num_files, 2):
            member1_index =  member_files[j].split('_')[1]
            member2_index = member_files[j+1].split('_')[1]
            assert member1_index == member2_index
            member1_path = archive_path.joinpath(member_files[j])
            member2_path = archive_path.joinpath(member_files[j+1])

            member1_bitmap = (np.load(member1_path) * 255).reshape((28,28))
            member2_bitmap = (np.load(member2_path) * 255).reshape((28,28))

            member1 = MNISTMember(bitmap_to_svg(member1_bitmap), 5)
            member2 = MNISTMember(bitmap_to_svg(member2_bitmap), 5)

            individual1 = MNISTIndividual(member1)
            individual2 = MNISTIndividual(member2)

            individual1.fitness = FakeFitness()
            individual2.fitness = FakeFitness()

            individual1.evaluate(problem)
            individual2.evaluate(problem)

            results[i].append((individual1.unsafe_region_probability, individual2.unsafe_region_probability))

            print(f'Individual {member1_index} evaluated ({j+1}/{num_files})')

    mnist_experiments_path.joinpath('probabilistic_eval.json').write_text(json.dumps(results))
