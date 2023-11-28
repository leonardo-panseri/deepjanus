import datetime
import json
import uuid
from collections import namedtuple
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt

from self_driving.beamng_roads import BeamNGRoad
from core.folders import FOLDERS, delete_folder_recursively

SimulationDataRecordProperties = ['timer', 'damage', 'pos', 'dir', 'vel', 'gforces', 'gforces2', 'steering',
                                  'steering_input', 'brake', 'brake_input', 'throttle', 'throttle_input',
                                  'throttleFactor', 'engineThrottle', 'wheelspeed', 'vel_kmh', 'is_oob', 'oob_counter',
                                  'max_oob_percentage', 'oob_distance', 'dist_from_goal']

SimulationDataRecord = namedtuple('SimulationDataRecord', SimulationDataRecordProperties)
SimulationDataRecords = List[SimulationDataRecord]

SimulationParams = namedtuple('SimulationParameters', ['iteration_steps', 'fps_limit'])


class SimulationInfo:

    start_time: str
    end_time: str
    success: bool
    exception_str: str
    computer_name: str
    ip_address: str
    id: str


class SimulationData:

    f_info = 'info'
    f_params = 'params'
    f_road = 'road'
    f_records = 'records'

    def __init__(self, simulation_name: str):
        self.name = simulation_name
        self.path_root: Path = FOLDERS.simulations.joinpath(simulation_name)
        self.path_json: Path = self.path_root.joinpath('simulation.full.json')
        self.path_partial: Path = self.path_root.joinpath('simulation.partial.tsv')
        self.path_road_img: Path = self.path_root.joinpath('road')
        self.id: str = None
        self.params: SimulationParams = None
        self.road: BeamNGRoad | None = None
        self.states: SimulationDataRecord = None
        self.info: SimulationInfo = None

        assert len(self.name) >= 3, 'the simulation name must be a string of at least 3 character'

    @property
    def n(self):
        return len(self.states)

    def set(self, params: SimulationParams, road: BeamNGRoad,
            states: SimulationDataRecords, info: SimulationInfo = None):
        self.params = params
        self.road = road
        if info:
            self.info = info
        else:
            self.info = SimulationInfo()
            self.info.id = str(uuid.uuid4())
        self.states = states

    def clean(self):
        delete_folder_recursively(self.path_root)

    def save(self):
        self.path_root.mkdir(parents=True, exist_ok=True)
        with open(self.path_json, 'w') as f:
            f.write(json.dumps({
                self.f_params: self.params._asdict(),
                self.f_info: self.info.__dict__,
                self.f_road: self.road.to_dict(),
                self.f_records: [r._asdict() for r in self.states]
            }))

        with open(self.path_partial, 'w') as f:
            sep = '\t'
            f.write(sep.join(SimulationDataRecordProperties) + '\n')
            gen = (r._asdict() for r in self.states)
            gen2 = (sep.join([str(d[key]) for key in SimulationDataRecordProperties]) + '\n' for d in gen)
            f.writelines(gen2)

        fig, ax = plt.subplots()
        self.road.to_image(ax)
        fig.savefig(self.path_road_img.with_suffix('.svg'))

    def load(self) -> 'SimulationData':
        with open(self.path_json, 'r') as f:
            obj = json.loads(f.read())
        info = SimulationInfo()

        info.__dict__ = obj.get(self.f_info, {})
        self.set(
            SimulationParams(**obj[self.f_params]),
            BeamNGRoad.from_dict(obj[self.f_road]),
            [SimulationDataRecord(**r) for r in obj[self.f_records]],
            info=info)
        return self

    def complete(self) -> bool:
        return self.path_json.exists()

    def min_oob_distance(self) -> float:
        return min(state.oob_distance for state in self.states)

    def start(self):
        self.info.success = None
        self.info.start_time = str(datetime.datetime.now())
        try:
            import platform
            self.info.computer_name = platform.node()
        except Exception as ex:
            self.info.computer_name = str(ex)

    def end(self, success: bool, exception=None):
        self.info.end_time = str(datetime.datetime.now())
        self.info.success = success
        if exception:
            self.exception_str = str(exception)


if __name__ == '__main__':
    for s in (sim.parts[-2:] for sim in FOLDERS.simulations.joinpath('beamng_nvidia_runner').glob('*')):
        sim1 = SimulationData('/'.join(s)).load()
        if len(sim1.states) == 0:
            print(sim1.name)
    # sim2 = SimulationData('group1/sim_road12_round2').load()
    # print(sim1.max_oob_distance())
    # print(sim2.max_oob_distance())
