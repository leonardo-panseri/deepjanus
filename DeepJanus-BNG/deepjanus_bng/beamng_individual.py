import json
import multiprocessing
import os
import threading
import timeit
from typing import TYPE_CHECKING

from deepjanus.individual import Individual
from deepjanus.log import get_logger
from .beamng_config import BeamNGConfig
from .beamng_interface import BeamNGInterface
from .beamng_member import BeamNGMember

if TYPE_CHECKING:
    from .beamng_problem import BeamNGProblem

log = get_logger(__file__)
# Need to have this here for parallel evaluations
PROJECT_ROOT = ""


class BeamNGIndividual(Individual[BeamNGMember]):
    """Individual for DeepJanus-BNG"""

    def __init__(self, mbr: BeamNGMember, seed_index: BeamNGMember = None,
                 neighbors: list[BeamNGMember] = None, name: str = None):
        """Creates a DeepJanus-BNG individual. Parameter 'name' can be passed to create clones of existing individuals,
        disabling the automatic incremental names."""
        super().__init__(mbr, seed_index, neighbors, name)
