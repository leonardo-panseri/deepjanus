from core.archive import Archive
from core.evaluator import Evaluator
from core.log import get_logger
from core.mutator import Mutator
from core.problem import Problem
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_evaluator import BeamNGLocalEvaluator
from self_driving.beamng_individual import BeamNGIndividual
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_mutator import BeamNGRoadMutator
from self_driving.shapely_roads import RoadGenerator

log = get_logger(__file__)


class BeamNGProblem(Problem):
    """Representation of the DeepJanus-BNG problem"""

    def __init__(self, config: BeamNGConfig, archive: Archive):
        self.config: BeamNGConfig = config
        super().__init__(config, archive)

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

    def generate_random_member(self, name: str = None) -> BeamNGMember:
        control_nodes, sample_nodes, gen_boundary = RoadGenerator(
            num_control_nodes=self.config.NUM_CONTROL_NODES,
            seg_length=self.config.SEG_LENGTH).generate()
        return BeamNGMember(control_nodes, sample_nodes, RoadGenerator.NUM_SPLINE_NODES, gen_boundary, name)

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
