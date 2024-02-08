from deepjanus.individual import Individual
from deepjanus.log import get_logger

from .mnist_member import MNISTMember

log = get_logger(__file__)


class MNISTIndividual(Individual[MNISTMember]):
    """Individual for DeepJanus-MNIST"""

    def __init__(self, mbr: MNISTMember, seed: MNISTMember = None,
                 neighbors: list[MNISTMember] = None, name: str = None):
        super().__init__(mbr, seed, neighbors, name)
