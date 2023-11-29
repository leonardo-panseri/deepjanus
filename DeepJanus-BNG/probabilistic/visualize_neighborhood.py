from typing import List, Optional

from core.folders import FOLDERS, FolderStorage
from self_driving.beamng_member import BeamNGMember

# Folder containing the serialized results of the experiment that we want to examine
EXPERIMENT_FOLDER = FOLDERS.experiments.joinpath('HQ_1')
INDIVIDUALS_INDEX = [4, 210, 321, 360, 422]


def find_mutated_nodes(mbr1: BeamNGMember, mbr2: BeamNGMember):
    """
    Finds the indexes of the road control nodes that have been mutated in the original individual
    :return: the indexes of the mutated nodes
    """
    mutated = []
    for k in range(len(mbr1.control_nodes)):
        if mbr1.control_nodes[k] != mbr2.control_nodes[k]:
            mutated.append(k)
    return mutated


def calculate_mutation_values(member1: BeamNGMember, member2: BeamNGMember):
    nodes = find_mutated_nodes(member1, member2)
    mut = {}
    for node in nodes:
        mut_x = member1.control_nodes[node][0] - member2.control_nodes[node][0]
        mut_y = member1.control_nodes[node][1] - member2.control_nodes[node][1]
        mut[node] = {'x': mut_x, 'y': mut_y}
    return mut


def pretty_print_neighborhood(member: BeamNGMember, neighborhood: List[BeamNGMember]):
    for i in range(len(neighborhood)):
        neighbor = neighborhood[i]
        mut = calculate_mutation_values(member, neighbor)
        print(f'    - Neighbor at index {i}')
        for n in mut:
            print(f'      Mutation for node {n}: {mut[n]}')

        outside_frontier: Optional[bool] = None
        if neighbor.distance_to_frontier:
            outside_frontier = True if neighbor.distance_to_frontier < 0 else False
        print(f'      Sim Result: {"NONE" if outside_frontier is None else "OUTSIDE" if outside_frontier else "INSIDE"}')


if __name__ == '__main__':
    nbh_storage = FolderStorage(EXPERIMENT_FOLDER.joinpath('neighbors'), 'ind{}.json')
    # ind_storage = FolderStorage(EXPERIMENT_FOLDER.joinpath('archive'), 'ind{}.json')

    for path in nbh_storage.all_files():
        nbh = nbh_storage.load_json_by_path(path)
        if nbh['original_individual_member_inside'] == "m1":
            mbr_in = BeamNGMember.from_dict(nbh['original_individual']['m1'])
            mbr_out = BeamNGMember.from_dict(nbh['original_individual']['m2'])
        else:
            mbr_in = BeamNGMember.from_dict(nbh['original_individual']['m2'])
            mbr_out = BeamNGMember.from_dict(nbh['original_individual']['m1'])
        print(f'Loaded individual {nbh["original_individual"]["name"]}')

        print(f'Original mutations: {calculate_mutation_values(mbr_in, mbr_out)}')

        print('Neighborhood of member inside the frontier:')
        nbh_in = [BeamNGMember.from_dict(n) for n in nbh['neighborhood_IN']]
        pretty_print_neighborhood(mbr_in, nbh_in)

        print('Neighborhood of member outside the frontier:')
        nbh_out = [BeamNGMember.from_dict(n) for n in nbh['neighborhood_OUT']]
        pretty_print_neighborhood(mbr_out, nbh_out)

        print()
