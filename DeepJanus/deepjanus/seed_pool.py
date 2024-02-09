import os.path
import random
from typing import TYPE_CHECKING

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
        raise NotImplemented()

    def __getitem__(self, item) -> Member:
        raise NotImplemented()

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
    """Seed pool that generates n random members and then accesses them sequentially"""

    def __init__(self, problem, n):
        super().__init__(problem)
        self.n = n
        self.seeds = [problem.generate_random_member(f"seed{i}") for i in range(self.n)]

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.seeds[item]


class SeedPoolFolder(SeedPool):
    """Seed pool that loads members from their serialized representation in a folder, either sequentially or randomly"""

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
