import json
import random
from json import JSONEncoder
from pathlib import Path

import numpy
from deap import tools

from .archive import Archive
from .config import Config, SeedPoolStrategy
from .evaluator import Evaluator
from .individual import Individual
from .log import get_logger
from .member import Member
from .metrics import calculate_seed_radius, calculate_diameter
from .mutator import Mutator
from .seed_pool import SeedPoolFolder, SeedPoolRandom

log = get_logger(__file__)


class Problem:
    """Class representing a problem to be solved by DeepJanus."""
    def __init__(self, config: Config):
        self.config: Config = config
        self.archive = self.initialize_archive()

        if self.config.SEED_POOL_STRATEGY == SeedPoolStrategy.GEN_RANDOM:
            self.seed_pool = SeedPoolRandom(self, config.POP_SIZE)
        elif self.config.SEED_POOL_STRATEGY == SeedPoolStrategy.GEN_RANDOM_SEEDED:
            self.seed_pool = SeedPoolFolder(self, False, config.SEED_FOLDER)
        elif self.config.SEED_POOL_STRATEGY == SeedPoolStrategy.GEN_SEQUENTIAL_SEEDED:
            self.seed_pool = SeedPoolFolder(self, True, config.SEED_FOLDER)
        else:
            raise ValueError(f"Seed pool strategy {self.config.SEED_POOL_STRATEGY} is invalid")

        self.individual_creator = None

        self._evaluator: Evaluator | None = None
        self._mutator: Mutator | None = None

        self.experiment_path = config.FOLDERS.experiments.joinpath(self.config.EXPERIMENT_NAME)

        self.current_generation_path: Path | None = None
        self.current_population_path: Path | None = None
        self.current_archive_path: Path | None = None

    def deap_individual_class(self):
        """Returns the class that represents individuals for this problem."""
        raise NotImplemented()

    def deap_generate_individual(self, seed_index: int = None) -> Individual:
        """Generates a new individual from the seed pool."""
        if seed_index is None:
            seed, seed_index = self.seed_pool.get_seed()
        else:
            seed = self.seed_pool[seed_index]

        # Need to use the DEAP creator to instantiate new individual
        individual: Individual = self.individual_creator(seed.clone(), seed_index)
        individual.mutate(self)

        log.info(f'Generated {individual}')
        return individual

    def deap_evaluate_individual(self, individual: Individual) -> tuple[float, float]:
        """Evaluates an individual of this problem."""
        fitness_values = individual.evaluate(self)
        return fitness_values

    def deap_mutate_individual(self, individual: Individual):
        """Mutates an individual of this problem."""
        individual.mutate(self)

    def deap_update_archive(self, population: list[Individual]):
        self.archive.process_population(population)

        for ind in self.archive:
            ind.save(self.current_archive_path)

    def deap_repopulate(self, population: list[Individual]) -> list[Individual]:
        """Repopulates by substituting individuals that are evolved from a seed that already generated
         a solution in the archive. Returns a list of new individuals that have been inserted in the population."""
        if len(self.archive) > 0:
            archived_seed_indices = [i.seed_index for i in self.archive]

            non_archived_seed_indices = []
            for i in range(len(self.seed_pool)):
                if i in archived_seed_indices:
                    continue
                non_archived_seed_indices.append(i)

            if len(non_archived_seed_indices) == 0:
                return []

            new_individuals = []
            for i in range(len(population)):
                if population[i].seed_index in archived_seed_indices:
                    seed_index = random.choice(non_archived_seed_indices)
                    new_individual = self.deap_generate_individual(seed_index)
                    log.info(f'Repopulation: substitute {population[i]} with {new_individual}')
                    population[i] = new_individual
                    new_individuals.append(new_individual)
            return new_individuals
        return []

    def member_class(self):
        """Returns the class that represents members for this problem."""
        raise NotImplemented()

    def generate_random_member(self, name: str = None) -> Member:
        """Generates a random member for this problem."""
        raise NotImplemented()

    def on_experiment_start(self):
        """Callback to execute actions at the start of the experiment."""
        config = dict(self.config.__dict__)
        del config['FOLDERS']
        self.experiment_path.joinpath('config.json').write_text(json.dumps(config))

    def on_experiment_end(self, logbook: tools.Logbook, tot_time: float):
        """Callback to execute actions at the end of the experiment."""
        report = {
            'generations': self.config.NUM_GENERATIONS,
            'archive_len': len(self.archive),
            'time': int(tot_time)
        }
        self.experiment_path.joinpath(f'report.json').write_text(json.dumps(report))
        self.experiment_path.joinpath(f'logbook.json').write_text(json.dumps(logbook, cls=NumpyArrayEncoder))

    def on_iteration_start(self, gen_idx: int):
        """Callback to execute actions at the start of each iteration."""
        self.current_generation_path = self.experiment_path.joinpath(f'gen{gen_idx}')

        self.current_population_path = self.current_generation_path.joinpath('population')
        self.current_population_path.mkdir(parents=True, exist_ok=True)

        self.current_archive_path = self.current_generation_path.joinpath('archive')
        self.current_archive_path.mkdir(parents=True, exist_ok=True)

    def on_iteration_end(self, gen_idx: int, stats_record: dict):
        """Callback to execute actions at the end of each iteration."""
        (self.current_generation_path.joinpath('fitness_stats.json')
         .write_text(json.dumps(stats_record, cls=NumpyArrayEncoder)))
        (self.experiment_path.joinpath('status.json')
         .write_text(json.dumps({'last_gen': gen_idx,
                                 'ind_counter': Individual.counter})))

    def initialize_archive(self) -> Archive:
        """Initializes the archive to store solutions of the problem."""
        raise NotImplemented()

    def get_evaluator(self) -> Evaluator:
        """Returns the evaluator that implements the strategy for evaluating members."""
        raise NotImplemented()

    def get_mutator(self) -> Mutator:
        """Returns the mutator that implements the strategy for mutating members."""
        raise NotImplemented()


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
