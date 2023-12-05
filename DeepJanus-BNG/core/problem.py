import json
from json import JSONEncoder
from pathlib import Path

import numpy
from deap import tools

from core.config import Config
from core.archive import Archive
from core.evaluator import Evaluator
from core.folders import FOLDERS, delete_folder_recursively
from core.individual import Individual
from core.log import get_logger
from core.member import Member
from core.metrics import calculate_seed_radius, calculate_diameter
from core.mutator import Mutator
from core.seed_pool import SeedPoolFolder, SeedPoolRandom

log = get_logger(__file__)


class Problem:
    """Class representing a problem to be solved by DeepJanus."""
    def __init__(self, config: Config, archive: Archive):
        self.config: Config = config
        self.archive = archive

        if self.config.SEED_POOL_STRATEGY == self.config.GEN_RANDOM:
            self.seed_pool = SeedPoolRandom(self, config.POP_SIZE)
        elif self.config.SEED_POOL_STRATEGY == self.config.GEN_RANDOM_SEEDED:
            self.seed_pool = SeedPoolFolder(self, False, config.SEED_FOLDER)
        elif self.config.SEED_POOL_STRATEGY == self.config.GEN_SEQUENTIAL_SEEDED:
            self.seed_pool = SeedPoolFolder(self, True, config.SEED_FOLDER)
        else:
            raise ValueError(f"Seed pool strategy {self.config.SEED_POOL_STRATEGY} is invalid")

        self.individual_creator = None

        self._evaluator: Evaluator | None = None
        self._mutator: Mutator | None = None

        self.experiment_path = FOLDERS.experiments.joinpath(self.config.EXPERIMENT_NAME)
        delete_folder_recursively(self.experiment_path)
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        self.current_generation_path: Path | None = None
        self.current_population_path: Path | None = None
        self.current_archive_path: Path | None = None

    def deap_individual_class(self):
        """Returns the class that represents individuals for this problem."""
        raise NotImplemented()

    def deap_generate_individual(self) -> Individual:
        """Generates a new individual from the seed pool."""
        seed = self.seed_pool.get_seed()

        # Need to use the DEAP creator to instantiate new individual
        individual: Individual = self.individual_creator(seed.clone(), seed)

        log.info(f'Generated {individual}')
        return individual

    def deap_evaluate_individual(self, individual: Individual) -> tuple[float, float]:
        """Evaluates an individual of this problem."""
        fitness_values = individual.evaluate(self)
        individual.save(self.current_population_path)
        return fitness_values

    def deap_mutate_individual(self, individual: Individual):
        """Mutates an individual of this problem."""
        individual.mutate(self)

    def deap_update_archive(self, population: list[Individual]):
        self.archive.process_population(population)

        for ind in self.archive:
            ind.save(self.current_archive_path)

    def deap_repopulate(self, population: list[Individual]):
        """Repopulates by substituting individuals that are evolved from a seed that already generated
         a solution in the archive."""
        if len(self.archive) > 0:
            archived_seeds = [i.seed for i in self.archive]
            for i in range(len(population)):
                if population[i].seed in archived_seeds:
                    # TODO: It should be at least mutated, and it should be generated from a non-archived seed
                    ind1 = self.deap_generate_individual()
                    log.info(f'Reseed rem {population[i]}')
                    population[i] = ind1

    def member_class(self):
        """Returns the class that represents members for this problem."""
        raise NotImplemented()

    def generate_random_member(self, name: str = None) -> Member:
        """Generates a random member for this problem."""
        raise NotImplemented()

    def on_experiment_start(self):
        """Callback to execute actions at the start of the experiment."""
        self.experiment_path.joinpath('config.json').write_text(json.dumps(self.config.__dict__))

    def on_experiment_end(self, logbook: tools.Logbook):
        """Callback to execute actions at the end of the experiment."""
        report = {
            'generations': self.config.NUM_GENERATIONS,
            'archive_len': len(self.archive),
            'radius': calculate_seed_radius(self.archive),
            'diameter': calculate_diameter(self.archive)
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

    def on_iteration_end(self, stats_record: dict):
        """Callback to execute actions at the end of each iteration."""
        (self.current_generation_path.joinpath('fitness_stats.json')
         .write_text(json.dumps(stats_record, cls=NumpyArrayEncoder)))

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
