import json

from beamng_config import BeamNGConfig
from core.folder_storage import SeedStorage
from self_driving.road_generator import RoadGenerator

if __name__ == "__main__":
    config = BeamNGConfig()
    seed_storage = SeedStorage(config.SEED_FOLDER)
    for i in range(1, 4):
        path = seed_storage.get_path_by_index(i)
        if path.exists():
            print('file ', path, 'already exists')
        else:
            obj = RoadGenerator(
                num_control_nodes=config.NUM_CONTROL_NODES,
                seg_length=config.SEG_LENGTH).generate()
            print('saving', path)
            path.write_text(json.dumps(obj.to_dict()))
