from typing import Iterable

from individual import Individual
from log import get_logger
from misc import closest_elements

log = get_logger(__file__)


class Archive(set):
    """Base class representing the archive of non-dominated individuals"""
    def process_population(self, pop: Iterable[Individual]):
        """
        Processes a new population to decide which individuals to store in the archive.
        :param pop: the population to process
        """
        raise NotImplemented()


class GreedyArchive(Archive):
    """Archive that stores every individual at the frontier"""
    def process_population(self, pop: Iterable[Individual]):
        for candidate in pop:
            if candidate.oob_ff < 0:
                self.add(candidate)


class SmartArchive(Archive):
    """Archive that stores only individuals that are distant at least a configurable threshold from each other"""
    def __init__(self, archive_threshold):
        super().__init__()
        self.ARCHIVE_THRESHOLD = archive_threshold

    def process_population(self, pop: Iterable[Individual]):
        for candidate in pop:
            assert candidate.oob_ff, candidate.name
            if candidate.oob_ff < 0:
                if len(self) == 0:
                    self._add(candidate)
                    log.debug('add initial individual')
                else:
                    # uses semantic_distance to exploit behavioral information
                    closest_archived, candidate_archived_distance = \
                        closest_elements(self, candidate, lambda a, b: a.semantic_distance(b))[0]
                    closest_archived: Individual

                    if candidate_archived_distance > self.ARCHIVE_THRESHOLD:
                        log.debug('candidate is far from any archived individual')
                        self._add(candidate)
                    else:
                        log.debug('candidate is very close to an archived individual')
                        if candidate.members_distance < closest_archived.members_distance:
                            log.debug('candidate is better than archived')
                            self._add(candidate)
                            self.remove(closest_archived)
                            log.info('archive rem ', closest_archived)
                        else:
                            log.debug('archived is better than candidate')

    def _add(self, candidate):
        self.add(candidate)
        log.info('archive add ', candidate)
