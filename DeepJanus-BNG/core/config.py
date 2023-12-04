class Config:
    """Class storing all DeepJanus parameters that can be tweaked during experiments."""

    # Seed pool strategy: create randomized individuals and use them as seeds
    GEN_RANDOM = 'GEN_RANDOM'
    # Seed pool strategy: pick random individuals from seed pool
    GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
    # Seed pool strategy: pick sequential individuals from seed pool
    GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'

    def __init__(self):
        # Unique identifier for the experiment
        self.EXPERIMENT_NAME = 'exp'
        # Weights to be multiplied to the fitness functions,
        # a positive value means that the function will be maximized, a negative value means that it will be minimized
        self.FITNESS_WEIGHTS = (1.0, -1.0)

        # Size of the population
        self.POP_SIZE = 12
        # Number of generations
        self.NUM_GENERATIONS = 150

        # Absolute value of bound for the mutation of a member
        self.MUTATION_EXTENT = 6.0
        # Minimum distance that an individual must have with all other individuals in the archive to be added
        self.ARCHIVE_THRESHOLD = 35.0

        # Unsafe region probability threshold
        self.PROBABILITY_THRESHOLD = 0.01
        # Desired confidence level for calculating the confidence interval
        self.CONFIDENCE_LEVEL = 0.90
        # Target error for deciding when to stop neighborhood exploration
        self.TARGET_ERROR = 0.1
        # TODO: Check if this is needed. We will have a fixed number of neighbors for all individuals given a target err
        # Maximum number of neighbors to generate for an individual
        self.MAX_NEIGHBORS = 20

        # Flag for turning on/off collection of extended data about simulations
        self.SAVE_SIM_DATA = True
        # Unique identifier for a simulation, can use '$(id)' for an incremental int identifier
        self.SIM_NAME = 'beamng_local_runner/sim_$(id)'

        # Type of seed pool strategy to utilize
        self.SEED_POOL_STRATEGY = Config.GEN_SEQUENTIAL_SEEDED
        # Name of the seed pool folder
        self.SEED_FOLDER = 'population_HQ1'
