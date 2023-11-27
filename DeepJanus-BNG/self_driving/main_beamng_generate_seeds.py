import json

from core.archive import SmartArchive
from core.folders import SeedStorage
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_problem import BeamNGProblem

config_silly = BeamNGConfig()
config_smart = BeamNGConfig()

problem_silly = BeamNGProblem(config_silly, SmartArchive(config_silly.ARCHIVE_THRESHOLD))
problem_smart = BeamNGProblem(config_smart, SmartArchive(config_smart.ARCHIVE_THRESHOLD))

if __name__ == '__main__':
    good_members_found = 0
    attempts = 0
    storage = SeedStorage('prova_roads')

    while good_members_found < 100:
        path = storage.get_path_by_index(good_members_found + 1)
        if path.exists():
            print('member already exists', path)
            good_members_found += 1
            continue
        attempts += 1
        print(f'attempts {attempts} good {good_members_found} looking for {path}')

        member = problem_silly.generate_random_member()
        member.evaluate(problem_silly.get_evaluator())
        if member.distance_to_frontier <= 0:
            continue

        member_smart = problem_smart.member_class().from_dict(member.to_dict())
        member_smart.clear_evaluation()
        member_smart.evaluate(problem_smart.get_evaluator())
        if member_smart.distance_to_frontier <= 0:
            continue

        member.distance_to_frontier = None
        good_members_found += 1
        path.write_text(json.dumps(member.to_dict()))
