import os

from deepjanus.config import Config, SeedPoolStrategy


class BeamNGConfig(Config):
    """Extension of the base DeepJanus config for BeamNG-specific parameters."""

    def __init__(self, project_root: str):
        super().__init__(project_root)
        self.PROJECT_ROOT = project_root

        # Unique identifier for the experiment
        self.EXPERIMENT_NAME = 'exp'
        # Weights to be multiplied to the fitness functions,
        # a positive value means that the function will be maximized, a negative value means that it will be minimized
        self.FITNESS_WEIGHTS = (1.0, -1.0)

        # Size of the population
        self.POP_SIZE = 12
        # Number of generations
        self.NUM_GENERATIONS = 100

        # Bounds for the mutation of a member
        self.MUTATION_LOWER_BOUND = 1
        self.MUTATION_UPPER_BOUND = 6
        # Flag indicating if the sign for the mutation value should be chosen randomly
        self.MUTATION_RANDOMIZE_SIGN = True
        # Minimum distance that an individual must have with all other individuals in the archive to be added
        self.ARCHIVE_THRESHOLD = 35.0

        # Unsafe region probability threshold
        self.PROBABILITY_THRESHOLD = 0.8
        # Desired confidence level for calculating the confidence interval
        self.CONFIDENCE_LEVEL = 0.90
        # Target error for deciding when to stop neighborhood exploration
        self.TARGET_ERROR = 0.1
        # Maximum number of neighbors to generate for an individual
        self.MAX_NEIGHBORS = 20

        # Number of parallel workers to use to evaluate a neighborhood
        # If < 2 evaluation will be sequential
        self.PARALLEL_EVALS = 11

        # Flag for turning on/off collection of extended data about simulations
        self.SAVE_SIM_DATA = False
        # Unique identifier for a simulation, can use '$(id)' for an incremental int identifier
        self.SIM_NAME = 'beamng_local_runner/sim_$(id)'

        # Type of seed pool strategy to utilize
        self.SEED_POOL_STRATEGY = SeedPoolStrategy.GEN_SEQUENTIAL_SEEDED
        # Name of the seed pool folder
        self.SEED_FOLDER = 'population_HQ1'

        # Number of control nodes for each road
        self.NUM_CONTROL_NODES = 10
        # Length of the interpolation segments to create a road from control nodes
        self.SEG_LENGTH = 25

        # Name of the Lane-Keeping Assist System Keras model
        self.MODEL_FILE = 'self-driving-car-185-2020'  # 'self-driving-car-4600'

        # Minimum speed that the car is allowed to drive (in km/h)
        self.MIN_SPEED = 10
        # Maximum speed that the car is allowed to drive (in km/h)
        self.MAX_SPEED = 25

        # BeamNG user data directory
        self.BEAMNG_USER_DIR = str(self.FOLDERS.simulations.joinpath('beamng', 'instance0', '0.31'))
        # Host for the BeamNG instance
        self.BEAMNG_HOST = 'localhost'
        # Port for the BeamNG instance
        self.BEAMNG_PORT = 65000
        # How many steps should the simulator advance at each iteration
        self.BEAMNG_STEPS = 5
        # How many frames should the simulator try to render in a second
        # This will influence the duration of a step: 1 step will be 1/fps
        self.BEAMNG_FPS = 30
        # How many simulations to run before restarting BeamNG, this can be useful to circumvent memory leaks
        # Set to -1 to disable
        self.BEAMNG_RESTART_AFTER = 20
