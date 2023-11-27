import json
import os
import re
import shutil
from pathlib import Path
from time import sleep

from core.log import get_logger

log = get_logger(__file__)


def delete_folder_recursively(path: str | Path):
    """Removes a folder and all its contents."""
    path = str(path)
    if not os.path.exists(path):
        return
    assert os.path.isdir(path), path
    log.info(f'Removing [{path}]')
    shutil.rmtree(path, ignore_errors=True)

    # sometimes rmtree fails to remove files
    for tries in range(20):
        if os.path.exists(path):
            sleep(0.1)
            shutil.rmtree(path, ignore_errors=True)

    if os.path.exists(path):
        shutil.rmtree(path)

    if os.path.exists(path):
        raise Exception(f'Unable to remove folder [{path}]')


class Folders:
    """Class containing paths to all folders needed by DeepJanus"""

    def __init__(self, core_folder: os.PathLike):
        """
        Initializes paths based on the location of core module.
        :param core_folder: path to the folder containing the DeepJanus core module
        """
        self.lib: Path = Path(core_folder).resolve()
        self.root: Path = self.lib.joinpath('..').resolve()
        self.data: Path = self.root.joinpath('data').absolute()
        self.log_ini: Path = self.data.joinpath('log.ini').absolute()
        self.member_seeds: Path = self.data.joinpath('member_seeds').absolute()
        self.experiments: Path = self.data.joinpath('experiments').absolute()
        self.simulations: Path = self.data.joinpath('simulations').absolute()
        self.trained_models_colab: Path = self.data.joinpath('trained_models_colab').absolute()


FOLDERS: Folders = Folders(os.path.dirname(__file__))


class FolderStorage:
    """Class for interfacing with a folder containing DeepJanus JSON serialized data"""

    def __init__(self, path: Path, mask: str):
        """
        Create an interface for storing and loading data in a folder. The folder will be created if it does not
        exist.
        :param path: the path to the folder
        :param mask: the pattern for file names in the folder (something like 'road{:03}_nodes.json')
        """
        self.mask = mask
        self.folder = path
        path.mkdir(parents=True, exist_ok=True)

    def all_files(self) -> list[str]:
        """Gets a list of absolute paths for all files contained in the folder sorted in alphabetical order."""

        def natural_keys(text):
            """Keys for natural ordering of file names containing digits"""
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

        expanded = [os.path.join(self.folder, filename) for filename in os.listdir(self.folder)]
        return sorted([path for path in expanded if os.path.isfile(path)], key=natural_keys)

    def get_path_by_index(self, index: int) -> Path:
        """Get the path to the file with the given index, using the mask to generate the file name."""
        assert index >= 0
        return self.folder.joinpath(self.mask.format(index))

    def load_json_by_index(self, index: int) -> dict:
        """Read and parse the JSON content of the file with the index, using the mask to generate the file name."""
        path = self.get_path_by_index(index)
        parsed = self.load_json_by_path(path)
        return parsed

    def save_json_by_index(self, index: int, object_instance):
        """Save a JSON representation of an object to the file with the index, using the mask to generate
         the file name."""
        path = self.get_path_by_index(index)
        dumps = self.save_json_by_path(path, object_instance)
        return dumps

    @classmethod
    def load_json_by_path(cls, path: str | Path):
        """Read and parse the JSON content of the file at the path."""
        assert os.path.exists(path), path
        with open(path, 'r') as f:
            nodes = json.loads(f.read())
        return nodes

    @classmethod
    def save_json_by_path(cls, path: str | Path, object_instance):
        """Save a JSON representation of an object to the file at the path."""
        with open(path, 'w') as f:
            dumps = json.dumps(object_instance)
            f.write(dumps)
        return dumps


class SeedStorage(FolderStorage):
    """Shorthand for creating a FolderStorage for a seed pool"""

    def __init__(self, subfolder: str):
        super().__init__(FOLDERS.member_seeds.joinpath(subfolder), 'seed{}.json')
