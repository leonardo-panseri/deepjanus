from pathlib import Path


def move_all_last_archives(system):
    experiments_path = Path(f'../DeepJanus-{system}/data/experiments')
    for experiment in experiments_path.glob('*/'):
        last_archive = experiment.glob('gen99/archive/*.json')
        new_path = Path(f'./{system}/{experiment.name}/archive').absolute()
        new_path.mkdir(parents=True, exist_ok=True)
        for ind in last_archive:
            ind.rename(new_path.joinpath(ind.name))


if __name__ == '__main__':
    move_all_last_archives('BNG')
    move_all_last_archives('MNIST')
