from typing import Iterable, Callable, TypeVar

import numpy as np

from .config import Config
from .individual import Individual
from .log import get_logger

log = get_logger(__file__)
T = TypeVar("T")


def closest_elements(elements_set: set[T], obj: T, distance_fun: Callable[[T, T], float]) -> list[tuple[T, float]]:
    """Returns pairs of closest elements."""
    elements = list(elements_set)
    distances = [distance_fun(obj, el) for el in elements]
    indexes = list(np.argsort(distances))
    result = [(elements[idx], distances[idx]) for idx in indexes]
    return result


class Archive(set):
    """Base class representing the archive of non-dominated individuals"""

    def __init__(self, config: Config):
        super().__init__()
        self.TARGET_ERROR = config.TARGET_ERROR

    def process_population(self, pop: Iterable[Individual]):
        """
        Processes a new population to decide which individuals to store in the archive.
        :param pop: the population to process
        """
        raise NotImplemented()

    def evaluate_sparseness(self, ind: Individual):
        """Calculates the minimum distance that the individual have from each other individual in a group."""
        elements = self - {ind}
        if len(elements) == 0:
            return 1.0

        # TODO: Should this be normalized?
        closest_element_dist = closest_elements(elements, ind, lambda a, b: a.distance(b))[0]
        return closest_element_dist[1]

    def find_candidates(self, population: Iterable[Individual]):
        candidates = []
        for individual in population:
            # TODO: Is this ok? Do we need a threshold for distance_to_frontier?
            error = (individual.distance_to_frontier[1] - individual.distance_to_frontier[0]) / 2.
            if error <= self.TARGET_ERROR:
                candidates.append(individual)
        return candidates


class GreedyArchive(Archive):
    """Archive that stores every individual at the frontier"""

    def process_population(self, pop: Iterable[Individual]):
        for candidate in self.find_candidates(pop):
            self.add(candidate)


class SmartArchive(Archive):
    """Archive that stores only individuals that are distant at least a configurable threshold from each other"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.ARCHIVE_THRESHOLD = config.ARCHIVE_THRESHOLD

    def process_population(self, pop: Iterable[Individual]):
        for candidate in self.find_candidates(pop):
            assert candidate.distance_to_frontier, candidate.name
            if len(self) == 0:
                self._add(candidate)
                log.debug('Add initial individual')
            else:
                # uses semantic_distance to exploit behavioral information
                closest_archived, candidate_archived_distance = \
                    closest_elements(self, candidate, lambda a, b: a.distance(b))[0]
                closest_archived: Individual

                if candidate_archived_distance > self.ARCHIVE_THRESHOLD:
                    log.debug('Candidate is far from any archived individual')
                    self._add(candidate)
                else:
                    log.debug('Candidate is very close to an archived individual')
                    # Compare fitness values for 'Distance to Frontier' and keep in the archive the closest individual
                    if candidate.fitness.values[1] < closest_archived.fitness.values[1]:
                        log.debug('Candidate is better than archived')
                        self._add(candidate)
                        self.remove(closest_archived)
                        log.info(f'Archive rem {closest_archived}')
                    else:
                        log.debug('Archived is better than candidate')

    def _add(self, candidate):
        self.add(candidate)
        log.info(f'Archive add {candidate}')
