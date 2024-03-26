from pathlib import Path
import json


def load_data(path, problem):
    data = {}

    data_path = Path(path)
    experiment_types = data_path.glob('*/')
    for experiment_type in experiment_types:
        if not experiment_type.is_dir() or str(experiment_type.name) == "seeds":
            continue

        type_data = {}

        experiments = experiment_type.glob('*/')
        for experiment in experiments:
            if not experiment.is_dir():
                continue

            experiment_data = {}

            with open(experiment.joinpath('report.json'), 'r') as f:
                report = json.load(f)
                experiment_data['time'] = report['time']

            archive = []
            solutions = experiment.glob('archive/*.json')
            for solution_path in solutions:
                with open(solution_path, 'r') as f:
                    solution = json.load(f)
                archive.append(problem.deap_individual_class().from_dict(solution, problem.deap_individual_class(),
                                                                         problem.member_class()))
            experiment_data['archive'] = archive

            type_data[str(experiment.name)] = experiment_data

        data[str(experiment_type.name)] = type_data
        print(f'Loaded {experiment_type.name} data')

    return data


def load_mnist_data():
    from deepjanus_mnist.mnist_problem import MNISTProblem
    from deepjanus_mnist.mnist_config import MNISTConfig

    path = './MNIST'
    cfg = MNISTConfig('../DeepJanus-MNIST')
    problem = MNISTProblem(cfg)

    return load_data(path, problem)


def load_beamng_data():
    from deepjanus_bng.beamng_problem import BeamNGProblem
    from deepjanus_bng.beamng_config import BeamNGConfig

    path = './BNG'
    cfg = BeamNGConfig('../DeepJanus-BNG')
    problem = BeamNGProblem(cfg)

    return load_data(path, problem)


if __name__ == '__main__':
    data = load_mnist_data()

    pairs = ['20_unsafe', '80_unsafe', '50_unsafe', '60_unsafe']
    for pair in pairs:
        for i in range(1, 11):
            exp_name = str(i)
            exp = data[pair][exp_name]

            archive_path = Path('MNIST', pair, exp_name, 'archive')
            for ind in exp['archive']:
                ind.save(archive_path)
