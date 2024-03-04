import os.path
import random
from typing import TYPE_CHECKING, Generator

from .folders import SeedStorage
from .log import get_logger
from .member import Member

if TYPE_CHECKING:
    from .problem import Problem

log = get_logger(__file__)


class SeedPool:
    """Base class for implementing seed pools"""

    def __init__(self, problem: 'Problem', sequential=True):
        """
        Creates a seed pool for a problem.
        :param problem: the representation of the problem that DeepJanus needs to solve
        :param sequential: a flag indicating if the pool should return seeds sequentially or randomly
        """
        self.problem = problem
        self.sequential = sequential
        self.counter = 0

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item) -> Member:
        raise NotImplementedError()

    def get_seed(self) -> Member:
        """Gets a seed from the pool. The behavior of this method depends on the sequential flag set
        when creating the seed pool."""
        if self.sequential:
            seed = self[self.counter]
            self.counter = (self.counter + 1) % len(self)
        else:
            seed = random.choice(self)
        return seed


class SeedPoolRandom(SeedPool):
    """Seed pool that generates random members. This is outside the definition of seeds given in the paper
    and should be used only for testing"""

    def __init__(self, problem, n):
        super().__init__(problem)
        self.n = n
        self.seeds = [problem.generate_random_member(f"seed{i}") for i in range(self.n)]

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.seeds[item]


class SeedPoolFolder(SeedPool):
    """Seed pool that loads members from their serialized representation in a folder"""

    def __init__(self, problem: 'Problem', sequential: bool, folder_name: str):
        super().__init__(problem, sequential)
        self.storage = SeedStorage(problem.config, folder_name)
        self.file_path_list = self.storage.all_files()
        self.cache: dict[str, Member] = {}

        if len(self.file_path_list) == 0:
            log.warning(f'Seed folder "{folder_name}" is empty')

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, item) -> Member:
        path = self.file_path_list[item]
        result: Member = self.cache.get(path, None)
        if not result:
            result = self.problem.member_class().from_dict(self.storage.load_json_by_path(path),
                                                           os.path.basename(path).replace(".json", ""))
            self.cache[path] = result
        result.problem = self.problem
        return result


class SeedFileGenerator:
    """Class for generating serialized representation of seeds and saving them in files"""

    def __init__(self, problems: list['Problem'], folder: str, member_generator: Generator[Member, None, None]):
        """
        Creates a seed file generator. A candidate member will be considered a seed only if it passes evaluation
        for all given problems.
        :param problems: a list of problems for which a member must pass evaluation to be considered a seed
        :param folder: the folder where to store generated seeds
        :param member_generator: a generator yielding members that are seed candidates
        """
        assert problems
        self.problems = problems
        self.folder = folder
        self.generator = member_generator

    def is_candidate_valid(self, member: Member):
        for problem in self.problems:
            member.clear_evaluation()
            satisfy_requirements = problem.get_evaluator().evaluate_member_sequential(member).satisfy_requirements
            if not satisfy_requirements:
                return False
        return True

    def generate_seeds(self, n: int):
        """Generates n seeds and save their serialized representation to the folder."""
        seeds_found = 0
        attempts = 0
        storage = SeedStorage(self.problems[0].config, self.folder)

        while seeds_found < n:
            seed_index = seeds_found

            path = storage.get_path_by_index(seed_index)
            if path.exists():
                log.info(f'Skipping seed{seed_index}: already generated')
                seeds_found += 1
                continue

            attempts += 1
            log.info(f'Total attempts: {attempts}; Found {seeds_found}/{n}; Looking for seed{seed_index}')

            candidate = next(self.generator, None)
            if candidate is None:
                log.error(f'Could not find {n} seeds: candidate generator exhausted')
                break

            if self.is_candidate_valid(candidate):
                candidate.clear_evaluation()
                storage.save_json_by_path(path, candidate.to_dict())
                seeds_found += 1
