from .folders import Folders


class SeedPoolStrategy:
    """Class containing possible strategies for accessing the seed pool"""
    # Create randomized individuals and use them as seeds
    GEN_RANDOM = 'GEN_RANDOM'
    # Pick random individuals from seed pool
    GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
    # Pick sequential individuals from seed pool
    GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'


class Config:
    """Abstract class defining all DeepJanus parameters that can be tweaked during experiments"""

    #########################################
    # This is an abstract class, each problem
    # definition should have its own config
    # that overrides all the fields and
    # calls the constructor.
    #########################################

    def __init__(self, project_root: str):
        self.FOLDERS = Folders(project_root)

    # Unique identifier for the experiment
    EXPERIMENT_NAME = None
    # Weights to be multiplied to the fitness functions,
    # a positive value means that the function will be maximized, a negative value means that it will be minimized
    FITNESS_WEIGHTS = None

    # Size of the population
    POP_SIZE = None
    # Number of generations
    NUM_GENERATIONS = None

    # Bounds for the mutation of a member
    MUTATION_LOWER_BOUND = None
    MUTATION_UPPER_BOUND = None
    # Flag indicating if the sign for the mutation value should be chosen randomly
    MUTATION_RANDOMIZE_SIGN = None
    # Minimum distance that an individual must have with all other individuals in the archive to be added
    ARCHIVE_THRESHOLD = None

    # Unsafe region probability threshold
    PROBABILITY_THRESHOLD = None
    # Desired confidence level for calculating the confidence interval
    CONFIDENCE_LEVEL = None
    # Target error for deciding when to stop neighborhood exploration
    TARGET_ERROR = None
    # Maximum number of neighbors to generate for an individual
    MAX_NEIGHBORS = None

    # Number of parallel workers to use to evaluate a neighborhood
    # If < 2 evaluation will be sequential
    PARALLEL_EVALS = 2

    # Flag for turning on/off collection of extended data about simulations
    SAVE_SIM_DATA = None
    # Unique identifier for a simulation, can use '$(id)' for an incremental int identifier
    SIM_NAME = None

    # Type of seed pool strategy to utilize
    SEED_POOL_STRATEGY = None
    # Name of the seed pool folder
    SEED_FOLDER = None

    def clone(self) -> 'Config':
        """Creates a copy of this config. Note that the FOLDERS parameter will not be set, it is needed to explicitly
        invoke __init__ with the correct path on the new instance."""
        copy = self.__class__('.')
        del copy.FOLDERS
        for param in filter(lambda key: 'A' <= key[0] <= 'Z', dir(Config)):
            setattr(copy, param, getattr(self, param))
        return copy

    @classmethod
    def from_dict(cls, d: dict, root_path: str):
        """Iterates through the dictionary and copy key-value pairs from it to a new instance of Config,
        that will be returned."""
        new_instance = cls(root_path)
        for key, value in d.items():
            setattr(new_instance, key, value)
        return new_instance

    @classmethod
    def from_file(cls, file_path: str, root_path: str):
        """Reads a serialized config from a json file and returns a new instance with all the key-value pairs copied
        from the serialized version."""
        import os
        import json
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                json_str = f.read()
                d = json.loads(json_str)
                return cls.from_dict(d, root_path)
