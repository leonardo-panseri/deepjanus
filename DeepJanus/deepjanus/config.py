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

    # Absolute value of bound for the mutation of a member
    MUTATION_EXTENT = None
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

    # Flag for turning on/off collection of extended data about simulations
    SAVE_SIM_DATA = None
    # Unique identifier for a simulation, can use '$(id)' for an incremental int identifier
    SIM_NAME = None

    # Type of seed pool strategy to utilize
    SEED_POOL_STRATEGY = None
    # Name of the seed pool folder
    SEED_FOLDER = None
