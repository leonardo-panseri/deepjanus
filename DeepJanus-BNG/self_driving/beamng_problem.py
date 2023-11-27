import json
from typing import List

from deap import creator

from core.archive import Archive
from core.evaluator import Evaluator
from core.folders import FOLDERS, delete_folder_recursively
from core.log import get_logger
from core.member import Member
from core.metrics import get_radius_seed, get_diameter
from core.mutator import Mutator
from core.problem import Problem
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_evaluator import BeamNGLocalEvaluator
from self_driving.beamng_individual import BeamNGIndividual
from self_driving.beamng_individual_set_store import BeamNGIndividualSetStore
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_mutator import BeamNGRoadMutator
from self_driving.road_generator import RoadGenerator

log = get_logger(__file__)


class BeamNGProblem(Problem):

    def __init__(self, config: BeamNGConfig, archive: Archive):
        self.config: BeamNGConfig = config
        super().__init__(config, archive)

        self.experiment_path = FOLDERS.experiments.joinpath(self.config.EXPERIMENT_NAME)
        delete_folder_recursively(self.experiment_path)

        self._evaluator: Evaluator | None = None
        self._mutator: Mutator | None = None

    def deap_individual_class(self):
        return BeamNGIndividual

    def deap_generate_individual(self):
        seed = self.seed_pool.get_seed()
        road1 = seed.clone()
        road2 = seed.clone()
        road2.mutate(self.get_mutator())

        # Need to use the DEAP creator to instantiate new individual
        individual: BeamNGIndividual = self.individual_creator(road1, road2, seed)

        log.info(f'generated {individual}')
        return individual

    def member_class(self):
        return BeamNGMember

    def generate_random_member(self, name: str = None) -> Member:
        result = RoadGenerator(num_control_nodes=self.config.NUM_CONTROL_NODES,
                               seg_length=self.config.SEG_LENGTH).generate(name=name)
        return result

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

    def get_evaluator(self) -> Evaluator:
        if not self._evaluator:
            ev_name = self.config.BEAMNG_EVALUATOR
            # if ev_name == BeamNGConfig.EVALUATOR_FAKE:
            #     from self_driving.beamng_evaluator_fake import BeamNGFakeEvaluator
            #     self._evaluator = BeamNGFakeEvaluator(self.config)
            if ev_name == BeamNGConfig.EVALUATOR_LOCAL_BEAMNG:
                self._evaluator = BeamNGLocalEvaluator(self.config)
            # elif ev_name == BeamNGConfig.EVALUATOR_REMOTE_BEAMNG:
            #     from self_driving.beamng_evaluator_remote import BeamNGRemoteEvaluator
            #     self._evaluator = BeamNGRemoteEvaluator(self.config)
            else:
                raise NotImplemented(self.config.BEAMNG_EVALUATOR)

        return self._evaluator

    def get_mutator(self) -> Mutator:
        if not self._mutator:
            self._mutator = BeamNGRoadMutator(-int(self.config.MUTATION_EXTENT), int(self.config.MUTATION_EXTENT))

        return self._mutator
