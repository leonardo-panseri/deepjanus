from deepjanus.archive import Archive
from deepjanus.evaluator import Evaluator
from deepjanus.log import get_logger
from deepjanus.mutator import Mutator
from deepjanus.problem import Problem
from .beamng_config import BeamNGConfig
from .beamng_evaluator import BeamNGLocalEvaluator
from .beamng_individual import BeamNGIndividual
from .beamng_member import BeamNGMember
from .beamng_mutator import BeamNGRoadMutator
from .shapely_roads import RoadGenerator

log = get_logger(__file__)


class BeamNGProblem(Problem):
    """Representation of the DeepJanus-BNG problem"""

    def __init__(self, config: BeamNGConfig, archive: Archive):
        self.config: BeamNGConfig = config
        super().__init__(config, archive)

    def deap_individual_class(self):
        return BeamNGIndividual

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
            self._mutator = BeamNGRoadMutator(self.config)

        return self._mutator
