import os

from core.config import Config


class BeamNGConfig(Config):
    """Extension of the base DeepJanus config for BeamNG-specific parameters."""

    EVALUATOR_LOCAL_BEAMNG = 'EVALUATOR_LOCAL_BEAMNG'

    def __init__(self):
        super().__init__()

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
        self.BEAMNG_USER_DIR = os.path.join(os.getenv('LOCALAPPDATA'), 'BeamNG.tech', '0.30')
        # Host for the BeamNG instance
        self.BEAMNG_HOST = 'localhost'
        # Port for the BeamNG instance
        self.BEAMNG_PORT = 12345
        # How many steps should the simulator advance at each iteration
        self.BEAMNG_STEPS = 5
        # How many frames should the simulator try to render in a second
        self.BEAMNG_FPS = 30
        # How many simulations to run before restarting BeamNG, this can be useful to circumvent memory leaks
        # Set to -1 to disable
        self.BEAMNG_RESTART_AFTER = 22
        # Type of evaluator that runs the BeamNG simulations
        self.BEAMNG_EVALUATOR = self.EVALUATOR_LOCAL_BEAMNG
