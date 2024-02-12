from deepjanus.config import Config, SeedPoolStrategy


class MNISTConfig(Config):
    def __init__(self, project_root: str):
        super().__init__(project_root)

        # Unique identifier for the experiment
        self.EXPERIMENT_NAME = 'exp'
        # Weights to be multiplied to the fitness functions,
        # a positive value means that the function will be maximized, a negative value means that it will be minimized
        self.FITNESS_WEIGHTS = (1.0, -1.0)

        # Size of the population
        self.POP_SIZE = 100
        # Number of generations
        self.NUM_GENERATIONS = 100

        # Bounds for the mutation of a member
        self.MUTATION_LOWER_BOUND = 1  # 0.01
        self.MUTATION_UPPER_BOUND = 6  # 0.6
        # Flag indicating if the sign for the mutation value should be chosen randomly
        self.MUTATION_RANDOMIZE_SIGN = True
        # Minimum distance that an individual must have with all other individuals in the archive to be added
        self.ARCHIVE_THRESHOLD = 4.0

        # Unsafe region probability threshold
        self.PROBABILITY_THRESHOLD = 0.01
        # Desired confidence level for calculating the confidence interval
        self.CONFIDENCE_LEVEL = 0.90
        # Target error for deciding when to stop neighborhood exploration
        self.TARGET_ERROR = 0.1
        # Maximum number of neighbors to generate for an individual
        self.MAX_NEIGHBORS = 20

        # Flag for turning on/off collection of extended data about simulations
        self.SAVE_SIM_DATA = False
        # Unique identifier for a simulation, can use '$(id)' for an incremental int identifier
        self.SIM_NAME = 'mnist_local_runner/sim_$(id)'

        # Type of seed pool strategy to utilize
        self.SEED_POOL_STRATEGY = SeedPoolStrategy.GEN_SEQUENTIAL_SEEDED
        # Name of the seed pool folder
        self.SEED_FOLDER = 'digit5'

        # ==================

        # Name of the handwritten digit classifier Keras model
        self.MODEL_FILE = 'cnnClassifier' # 'cnnClassifier_lowLR'