from matplotlib import pyplot as plt

from core import log, nsga2
from core.archive import SmartArchive
from core.folders import FOLDERS
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_problem import BeamNGProblem

config = BeamNGConfig()

problem = BeamNGProblem(config, SmartArchive(config.ARCHIVE_THRESHOLD))

if __name__ == '__main__':
    log.configure(FOLDERS.log_ini)

    nsga2.main(problem)
    print('done')

    plt.ioff()
    plt.show()
