import itertools
import json
import random
from typing import List

from deap import creator

from core.archive import Archive
from core.folders import FOLDERS, delete_folder_recursively
from core.log import get_logger
from core.member import Member
from core.metrics import get_radius_seed, get_diameter
from core.problem import Problem
from core.seed_pool import SeedPoolAccessStrategy, SeedPoolRandom, SeedPoolFolder
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_evaluator import BeamNGEvaluator
from self_driving.beamng_individual import BeamNGIndividual
from self_driving.beamng_individual_set_store import BeamNGIndividualSetStore
from self_driving.beamng_member import BeamNGMember
from self_driving.road_generator import RoadGenerator

log = get_logger(__file__)


class BeamNGProblem(Problem):
    def __init__(self, config: BeamNGConfig, archive: Archive):
        self.config: BeamNGConfig = config
        self._evaluator: BeamNGEvaluator = None
        super().__init__(config, archive)
        if self.config.SEED_POOL_STRATEGY == self.config.GEN_RANDOM:
            seed_pool = SeedPoolRandom(self, config.POP_SIZE)
        else:
            seed_pool = SeedPoolFolder(self, config.SEED_FOLDER)
        self._seed_pool_strategy = SeedPoolAccessStrategy(seed_pool)
        self.experiment_path = FOLDERS.experiments.joinpath(self.config.EXPERIMENT_NAME)
        delete_folder_recursively(self.experiment_path)

    def deap_generate_individual(self):
        seed = self._seed_pool_strategy.get_seed()
        road1 = seed.clone()
        road2 = seed.clone().mutate()
        road1.config = self.config
        road2.config = self.config
        individual: BeamNGIndividual = creator.Individual(road1, road2, self.config, self.archive)
        individual.seed = seed
        log.info(f'generated {individual}')

        return individual

    def deap_evaluate_individual(self, individual: BeamNGIndividual):
        return individual.evaluate()

    def on_iteration(self, idx, pop: List[BeamNGIndividual], logbook):
        # self.archive.process_population(pop)

        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self.experiment_path.joinpath('config.json').write_text(json.dumps(self.config.__dict__))

        gen_path = self.experiment_path.joinpath(f'gen{idx}')
        gen_path.mkdir(parents=True, exist_ok=True)

        # Generate final report at the end of the last iteration.
        if idx + 1 == self.config.NUM_GENERATIONS:
            report = {
                'archive_len': len(self.archive),
                'radius': get_radius_seed(self.archive),
                'diameter_out': get_diameter([ind.members_by_sign()[0] for ind in self.archive]),
                'diameter_in': get_diameter([ind.members_by_sign()[1] for ind in self.archive])
            }
            gen_path.joinpath(f'report{idx}.json').write_text(json.dumps(report))

        BeamNGIndividualSetStore(gen_path.joinpath('population')).save(pop)
        BeamNGIndividualSetStore(gen_path.joinpath('archive')).save(self.archive)

    def generate_random_member(self) -> Member:
        result = RoadGenerator(num_control_nodes=self.config.NUM_CONTROL_NODES,
                               seg_length=self.config.SEG_LENGTH).generate()
        result.config = self.config
        result.problem = self
        return result

    def deap_individual_class(self):
        return BeamNGIndividual

    def member_class(self):
        return BeamNGMember

    def reseed(self, pop, offspring):
        if len(self.archive) > 0:
            archived_seeds = [i.seed for i in self.archive]
            for i in range(len(pop)):
                if pop[i].seed in archived_seeds:
                    ind1 = self.deap_generate_individual()
                    log.info(f'reseed rem {pop[i]}')
                    pop[i] = ind1

    def get_evaluator(self):
        if self._evaluator:
            return self._evaluator
        ev_name = self.config.BEAMNG_EVALUATOR
        # if ev_name == BeamNGConfig.EVALUATOR_FAKE:
        #     from self_driving.beamng_evaluator_fake import BeamNGFakeEvaluator
        #     self._evaluator = BeamNGFakeEvaluator(self.config)
        if ev_name == BeamNGConfig.EVALUATOR_LOCAL_BEAMNG:
            from self_driving.beamng_nvidia_runner import BeamNGNvidiaOob
            self._evaluator = BeamNGNvidiaOob(self.config)
        # elif ev_name == BeamNGConfig.EVALUATOR_REMOTE_BEAMNG:
        #     from self_driving.beamng_evaluator_remote import BeamNGRemoteEvaluator
        #     self._evaluator = BeamNGRemoteEvaluator(self.config)
        else:
            raise NotImplemented(self.config.BEAMNG_EVALUATOR)

        return self._evaluator

    def pre_evaluate_members(self, individuals: List[BeamNGIndividual]):
        # return
        # the following code does not work as wanted or expected!
        all_members = list(itertools.chain(*[(ind.m1, ind.m2) for ind in individuals]))
        log.info('----evaluation warmup')
        self.get_evaluator().evaluate(all_members)
        log.info('----warmpup completed')
