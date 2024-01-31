import json
import os
import shutil

from deepjanus.folders import delete_folder_recursively
from deepjanus.log import get_logger

# Workaround for keeping type hinting while avoiding circular imports
from typing import TYPE_CHECKING

from self_driving.beamng_config import BeamNGConfig

if TYPE_CHECKING:
    from self_driving.beamng_roads import BeamNGRoad

log = get_logger(__file__)

LEVEL_NAME = 'tig'


class MapFolder:
    """Interface for the folder containing a BeamNG TIG map"""

    def __init__(self, path):
        self.path = path
        self.version_path = os.path.join(path, 'tig-version.json')

    def exists(self):
        """Checks if the map folder exists."""
        return os.path.exists(self.path)

    def get_version(self):
        """Gets the version of the map in the folder."""
        if not os.path.exists(self.version_path):
            return None

        with open(self.version_path, 'r') as f:
            return json.load(f)['version']

    def delete(self):
        """Deletes the map folder."""
        delete_folder_recursively(self.path)

    def install_road(self, road: 'BeamNGRoad'):
        """Sets up the road to be simulated in the map."""
        with open(os.path.join(self.path, 'main/MissionGroup/generated/items.level.json'), 'w') as f:
            f.write(road.to_json() + '\n' + road.waypoint_goal.to_json())


class LevelsFolder:
    """Interface for the BeamNG levels folder"""

    def __init__(self, path):
        self.path = os.path.realpath(path)

    def get_map(self, map_name: str):
        """Gets the interface for a map folder."""
        return MapFolder(os.path.join(self.path, map_name))


class MapUtils:
    """Utilities to install the simulation map and set up roads"""

    def __init__(self):
        self.installed_levels = None
        local_levels_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'levels_template')
        self.local_levels = LevelsFolder(local_levels_path)

        self.local_map = self.local_levels.get_map('tig')
        self.installed_map = None

    def install_map_if_needed(self, bng_user_path: str):
        """Checks if the latest version of the map is installed. If not tries to install it."""
        bng_levels_path = os.path.join(bng_user_path, 'levels')
        os.makedirs(bng_levels_path, exist_ok=True)
        self.installed_levels = LevelsFolder(bng_levels_path)
        self.installed_map = self.installed_levels.get_map('tig')

        if self.installed_map.exists():
            installed_ver = self.installed_map.get_version()
            local_ver = self.local_map.get_version()
            if installed_ver is None:
                log.error(f"The folder [{self.installed_map.path}] does not look like a map of tig project: it does "
                          f"not contain the version file 'tig-version.json'")
                exit(1)
            else:
                if installed_ver != local_ver:
                    log.info(f'Maps have different version information. '
                             f'Do you want to remove all {self.installed_map.path} folder and copy it anew? '
                             f'Type y to accept, n to keep it as it is')
                    while True:
                        resp = input('>')
                        if resp in ['y', 'n']:
                            break
                        log.info('Type y or n')
                    if resp == 'y':
                        self.installed_map.delete()

        if not self.installed_map.exists():
            log.info(f'Copying map from [{self.local_map.path}] to [{self.installed_map.path}]')
            shutil.copytree(src=self.local_map.path, dst=self.installed_map.path)

    def install_road(self, road: 'BeamNGRoad'):
        """Sets up the road for a simulation."""
        if not self.installed_map or not self.installed_map.exists():
            raise f"Map is not installed"
        self.installed_map.install_road(road)


map_utils = MapUtils()

if __name__ == '__main__':
    cfg = BeamNGConfig(os.path.dirname(os.path.dirname(__file__)))
    map_utils.install_map_if_needed(cfg.BEAMNG_USER_DIR)
