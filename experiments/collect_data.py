from pathlib import Path


def move_all_experiments_data(system):
    experiments_path = Path(f'../DeepJanus-{system}/data/experiments')
    for experiment in experiments_path.glob('*/'):
        if not experiment.is_dir():
            continue

        last_archive = experiment.glob('gen99/archive/*.json')
        files = [experiment.joinpath('config.json'), experiment.joinpath('logbook.json'), experiment.joinpath('report.json')]

        exp_name = experiment.name.split('_')[0] + '_unsafe'
        exp_idx = experiment.name.split('_')[1]
        new_path = Path(f'./{system}/{exp_name}/{exp_idx}').absolute()
        new_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            file.rename(new_path.joinpath(file.name))
        for ind in last_archive:
            ind.rename(new_path.joinpath('archive', ind.name))


if __name__ == '__main__':
    move_all_experiments_data('BNG')
    move_all_experiments_data('MNIST')
