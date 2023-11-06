from enum import Enum, auto
from typing import Optional
import json

from core.archive_impl import SmartArchive
from core.folder_storage import FolderStorage
from core.folders import folders
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_problem import BeamNGProblem
from self_driving.beamng_individual import BeamNGIndividual

# Folder containing the serialized results of the experiment that we want to examine
EXPERIMENT_FOLDER = folders.experiments.joinpath('HQ_1')
# Indexes of the individuals that we want to examine
INDIVIDUALS_INDEX = [360, 422]
# Number of neighbors to generate and simulate for each individual
NEIGHBORHOOD_SIZE = 10


def load_individual(storage: FolderStorage, individual_index: int, problem: BeamNGProblem):
    """
    Loads an individual from a storage folder and prepares it for evaluation.
    :param storage: the FolderStorage where the serialized individuals are kept
    :param individual_index: the index of the individual to load
    :param problem: the BeamNGProblem that will be used for evaluation
    :return: a tuple containing the individual, the member inside the frontier and the member outside of it
    """
    # Load individual from its JSON representation in the storage
    individual = BeamNGIndividual.from_dict(storage.load_json_by_index(individual_index))
    individual.config = problem.config
    individual.archive = problem.archive

    # Check which member is inside the frontier and which is outside
    # Assuming that we have only individuals at the frontier in the storage
    if individual.m1.distance_to_boundary < 0:
        member_inside = individual.m2
        member_outside = individual.m1
    else:
        member_inside = individual.m1
        member_outside = individual.m2

    # Set distance to boundary for both members to None so that they can be re-evaluated
    member_inside.clear_evaluation()
    member_inside.problem = problem
    member_outside.clear_evaluation()
    member_outside.problem = problem

    return individual, member_inside, member_outside


class ComparisonResult(str, Enum):
    """Represents the possible outcomes of a comparison between two evaluations of an individual"""
    SAME = 'same'
    BOTH_OUTSIDE = 'both_outside'
    BOTH_INSIDE = 'both_inside'
    SWITCHED = 'switched'


def compare_individual_with_original(member_inside: BeamNGMember, member_outside: BeamNGMember):
    """
    Compares the evaluation of an individual on the frontier with a previous one.
    :param member_inside: the member that was inside the frontier in the original individual
    :param member_outside: the member that was outside the frontier in the original individual
    :return: a ComparisonResult value representing the outcome
    """
    result = ComparisonResult.SAME
    if member_inside.distance_to_boundary < 0 < member_outside.distance_to_boundary:
        result = ComparisonResult.SWITCHED
    elif member_inside.distance_to_boundary > 0 and member_outside.distance_to_boundary > 0:
        result = ComparisonResult.BOTH_INSIDE
    elif member_inside.distance_to_boundary < 0 and member_outside.distance_to_boundary < 0:
        result = ComparisonResult.BOTH_OUTSIDE

    return result


def generate_neighborhood(member_inside: BeamNGMember, member_outside: BeamNGMember, size: int):
    """
    Generates a neighborhood of an individual by creating new members that have the same road control node mutated
    with a different value respect to the original members of the individual.
    :param member_inside: the original member of the individual which is inside the frontier
    :param member_outside: the original member of the individual which is outside the frontier
    :param size: the size of the neighborhood to be generated
    :return: a list of new members that are neighbors of the individuals
    :raise Exception if a neighborhood of this size cannot be generated
    """
    def find_mutated_node():
        """
        Finds the index of the road control node that has been mutated in the original individual
        :return: the index of the mutated node, or None if it is not found
        """
        for k in range(len(member_inside.control_nodes)):
            if member_inside.control_nodes[k] != member_outside.control_nodes[k]:
                return k
        return None

    neighborhood = []
    MAX_ATTEMPTS = 50
    for _ in range(size):
        same_road = True
        new_mbr: Optional[BeamNGMember] = None
        # Try generating a new member that is not equal to any other in the neighborhood
        # or to the two original ones
        attempts = 0
        while same_road and attempts < MAX_ATTEMPTS:
            # Start from the original member inside the frontier
            new_mbr = member_inside.clone()

            # Find which control node was mutated in the original individual and mutate it again
            mutated_node = find_mutated_node()
            if not mutated_node:
                raise Exception('Cannot find mutated road control node')
            new_mbr.mutate(mutated_node)

            # Check if the new road obtained through mutation is equal to another one in the neighborhood
            # or in the members of the original individual
            same_road = False
            for neighbor in neighborhood + [member_inside, member_outside]:
                if neighbor.control_nodes == new_mbr.control_nodes:
                    same_road = True

            attempts += 1

        if attempts == MAX_ATTEMPTS:
            raise Exception(f'Cannot generate neighborhood of size {size}')

        neighborhood.append(new_mbr)
    return neighborhood


if __name__ == '__main__':
    # Set up the problem, needed for holding references to config, archive and evaluator
    cfg = BeamNGConfig()
    prob = BeamNGProblem(cfg, SmartArchive(cfg.ARCHIVE_THRESHOLD))

    # Folder containing the archive that we want to examine
    ind_storage = FolderStorage(EXPERIMENT_FOLDER.joinpath('archive'), 'ind{}.json')
    # Create the neighborhood exploration results folder if necessary
    neighborhood_folder = EXPERIMENT_FOLDER.joinpath('neighbors')
    neighborhood_folder.mkdir(exist_ok=True)

    # For each individual that we want to examine, re-run simulations of its original members
    # and then generate and simulate neighbors to estimate the chance of being outside the frontier
    for i in INDIVIDUALS_INDEX:
        print("==================== Evaluating original individual ====================")
        # Load the individual from file and evaluate it again
        ind, mbr_in, mbr_out = load_individual(ind_storage, i, prob)
        ind.evaluate()

        # Check if the evaluation results have changed compared to what was saved on file
        comp_w_original = compare_individual_with_original(mbr_in, mbr_out)

        # Generate members in the neighborhood
        nbh = generate_neighborhood(mbr_in, mbr_out, NEIGHBORHOOD_SIZE)
        outside_bound = 0
        print("==================== Evaluating neighborhood ====================")
        # For each member, evaluate it and check if it is outside the frontier
        for mbr in nbh:
            mbr.evaluate()

            if mbr.distance_to_boundary < 0:
                outside_bound += 1

        # Prepare the results of the neighborhood exploration for serialization
        out = {'original_individual': ind.to_dict(),
               'neighborhood_size': NEIGHBORHOOD_SIZE,
               'reevaluation_results': comp_w_original,
               'neighbors_outside_frontier_percentage': outside_bound / NEIGHBORHOOD_SIZE,
               'neighborhood': [mbr.to_dict() for mbr in nbh]}

        # Save the results to file
        with open(EXPERIMENT_FOLDER.joinpath('neighbors', f'ind{i}.json'), 'w') as f:
            f.write(json.dumps(out))
