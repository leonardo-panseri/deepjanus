from typing import Iterable, Callable, TypeVar

import numpy as np

from core.individual import Individual
from core.log import get_logger

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

        closest_element_dist = closest_elements(elements, ind, lambda a, b: a.distance(b))[0]
        return closest_element_dist[1]


class GreedyArchive(Archive):
    """Archive that stores every individual at the frontier"""

    def process_population(self, pop: Iterable[Individual]):
        for candidate in pop:
            # TODO: Check if a threshold for distance_to_frontier is needed for considering adding a candidate
            self.add(candidate)


class SmartArchive(Archive):
    """Archive that stores only individuals that are distant at least a configurable threshold from each other"""

    def __init__(self, archive_threshold):
        super().__init__()
        self.ARCHIVE_THRESHOLD = archive_threshold

    def process_population(self, pop: Iterable[Individual]):
        for candidate in pop:
            assert candidate.distance_to_frontier, candidate.name
            # TODO: Check if a threshold for distance_to_frontier is needed for considering adding a candidate
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
