from deepjanus.archive import Archive
from deepjanus.evaluator import Evaluator
from deepjanus.log import get_logger
from deepjanus.mutator import Mutator
from deepjanus.problem import Problem

from .mnist_config import MNISTConfig
from .mnist_evaluator import MNISTLocalEvaluator
from .mnist_individual import MNISTIndividual
from .mnist_member import MNISTMember
from .mnist_mutator import MNISTDigitMutator

log = get_logger(__file__)


class MNISTProblem(Problem):
    """Representation of the DeepJanus-MNIST problem"""

    def __init__(self, config: MNISTConfig, archive: Archive):
        self.config: MNISTConfig = config
        super().__init__(config, archive)

    def deap_individual_class(self):
        return MNISTIndividual

    def member_class(self):
        return MNISTMember

    def generate_random_member(self, name: str = None) -> MNISTMember:
        raise Exception('Random member generation is not supported by DeepJanus-MNIST')

    def get_evaluator(self) -> Evaluator:
        if not self._evaluator:
            self._evaluator = MNISTLocalEvaluator(self.config)

        return self._evaluator

    def get_mutator(self) -> Mutator:
        if not self._mutator:
            self._mutator = MNISTDigitMutator(self.config.MUTATION_LOWER_BOUND, self.config.MUTATION_EXTENT)

        return self._mutator