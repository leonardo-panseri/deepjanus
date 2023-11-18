import argparse
import json
import logging
import timeit
from datetime import datetime
from enum import Enum
from typing import Optional, List

from core.archive_impl import SmartArchive
from core.folder_storage import FolderStorage
from core.folders import folders
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_individual import BeamNGIndividual
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_problem import BeamNGProblem

# Folder containing the serialized results of the experiment that we want to examine
EXPERIMENT_FOLDER = folders.experiments.joinpath('HQ_1')
# Number of neighbors to generate and simulate for each individual
NEIGHBORHOOD_SIZE = 10


def load_individual(storage: FolderStorage, individual_index: int, problem: BeamNGProblem):
    """
    Loads an individual from a storage folder and prepares it for evaluation.
    :param storage: the FolderStorage where the serialized individuals are kept
    :param individual_index: the index of the individual to load
    :param problem: the BeamNGProblem that will be used for evaluation
    :return: a tuple containing the individual, the member inside the frontier, the member outside of it,
             and a flag indicating if m1 is the member inside (otherwise it is m2)
    """
    # Load individual from its JSON representation in the storage
    individual = BeamNGIndividual.from_dict(storage.load_json_by_index(individual_index))
    individual.config = problem.config
    individual.archive = problem.archive

    # Check which member is inside the frontier and which is outside
    # Assuming that we have only individuals at the frontier in the storage
    m1_inside: bool
    if individual.m1.distance_to_boundary < 0:
        member_inside = individual.m2
        member_outside = individual.m1
        m1_inside = False
    else:
        member_inside = individual.m1
        member_outside = individual.m2
        m1_inside = True

    # Set distance to boundary for both members to None so that they can be re-evaluated
    member_inside.clear_evaluation()
    member_inside.problem = problem
    member_outside.clear_evaluation()
    member_outside.problem = problem

    return individual, member_inside, member_outside, m1_inside


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
    :return: two lists of new members that are neighbors of the two members of the individual
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

    MAX_ATTEMPTS = 50

    def gen_mutation(member: BeamNGMember, neighborhood: List[BeamNGMember], other_mutations: dict):
        same_road = True
        new_mbr: Optional[BeamNGMember] = None
        # Try generating a new member that is not equal to any other in the neighborhood
        # or to the two original ones
        attempts = 0
        while same_road:
            # Start from the original member inside the frontier
            new_mbr = member.clone()

            new_mbr.mutate()

            # Check if the new road obtained through mutation is equal to another one in the neighborhood
            # or in the members of the original individual
            same_road = new_mbr.hex_hash() in other_mutations

            attempts += 1
            if attempts == MAX_ATTEMPTS:
                raise Exception(f'Cannot generate neighborhood of size {size}')

        other_mutations[new_mbr.hex_hash()] = True
        neighborhood.append(new_mbr)

    neighborhood_in = []
    neighborhood_out = []
    all_members = {member_inside.hex_hash(): True, member_outside.hex_hash(): True}
    for _ in range(size):
        gen_mutation(member_inside, neighborhood_in, all_members)
        gen_mutation(member_outside, neighborhood_out, all_members)

    return neighborhood_in, neighborhood_out


def explore_neighborhood(individuals: List[int]):
    # Set up the problem, needed for holding references to config, archive and evaluator
    cfg = BeamNGConfig()
    prob = BeamNGProblem(cfg, SmartArchive(cfg.ARCHIVE_THRESHOLD))

    # Set up logger
    log = logging.getLogger('ExploreNeighborhood')
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(EXPERIMENT_FOLDER
                             .joinpath(f'neighbors_{datetime.now().strftime("%d-%m_%H-%M")}.log'),
                             'w', 'utf8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    log.addHandler(fh)

    # Folder containing the archive that we want to examine
    ind_storage = FolderStorage(EXPERIMENT_FOLDER.joinpath('archive'), 'ind{}.json')
    # Create the neighborhood exploration results folder if necessary
    neighborhood_folder = EXPERIMENT_FOLDER.joinpath('neighbors')
    neighborhood_folder.mkdir(exist_ok=True)

    # For each individual that we want to examine, re-run simulations of its original members
    # and then generate and simulate neighbors to estimate the chance of being outside the frontier
    tot_start = timeit.default_timer()
    ind_times = {}
    for i in individuals:
        log.info(f'Starting evaluation of individual {i}')
        ind_start = timeit.default_timer()
        # Load the individual from file and evaluate it again
        ind, mbr_in, mbr_out, m1_in = load_individual(ind_storage, i, prob)
        ind.evaluate()

        # Check if the evaluation results have changed compared to what was saved on file
        comp_w_original = compare_individual_with_original(mbr_in, mbr_out)

        # Generate members in the neighborhood
        gen_nbh_start = timeit.default_timer()
        nbh_in, nbh_out = generate_neighborhood(mbr_in, mbr_out, NEIGHBORHOOD_SIZE)
        gen_nbh_time = timeit.default_timer() - gen_nbh_start
        log.info(f"Neighborhood generated in {gen_nbh_time}s")

        outside_frontier_in = 0
        log.info("Evaluating neighborhood of member inside the frontier...")
        # For each neighbor of the original member inside the frontier, evaluate it and check if it is outside
        for mbr_idx in range(len(nbh_in)):
            print(f"==================> Evaluating neighbor {mbr_idx} of member inside the frontier of individual {i}")
            mbr = nbh_in[mbr_idx]
            mbr.evaluate()

            if mbr.distance_to_boundary < 0:
                outside_frontier_in += 1

        outside_frontier_out = 0
        log.info("Evaluating neighborhood of member outside the frontier...")
        # For each neighbor of the original member outside the frontier, evaluate it and check if it is outside
        for mbr_idx in range(len(nbh_out)):
            print(f"==================> Evaluating neighbor {mbr_idx} of member outside the frontier of individual {i}")
            mbr = nbh_out[mbr_idx]
            mbr.evaluate()

            if mbr.distance_to_boundary < 0:
                outside_frontier_out += 1

        # Calculate simulation time for this individual
        ind_time = timeit.default_timer() - ind_start
        log.info(f"Evaluation completed in {ind_time}s")
        ind_times[i] = ind_time

        # Prepare the results of the neighborhood exploration for serialization
        out = {'neighborhood_size': NEIGHBORHOOD_SIZE,
               'reevaluation_results': comp_w_original,
               'neighbors_IN_outside_frontier_percentage': outside_frontier_in / NEIGHBORHOOD_SIZE,
               'neighbors_OUT_outside_frontier_percentage': outside_frontier_out / NEIGHBORHOOD_SIZE,
               'simulation_time': ind_time,
               'original_individual_member_inside': 'm1' if m1_in else 'm2',
               'original_individual': ind.to_dict(),
               'neighborhood_IN': [mbr.to_dict() for mbr in nbh_in],
               'neighborhood_OUT': [mbr.to_dict() for mbr in nbh_out]}

        # Save the results to file
        with open(EXPERIMENT_FOLDER.joinpath('neighbors', f'ind{i}.json'), 'w') as f:
            f.write(json.dumps(out))

    tot_time = timeit.default_timer() - tot_start
    log.info("Experiment COMPLETE")
    log.info(f"Total execution time: {tot_time}s")
    log.info(f"Individuals execution time: {ind_times}")


if __name__ == '__main__':
    # Get indexes of individuals which neighbourhood to explore from command line arguments
    parser = argparse.ArgumentParser(description='Explore the neighborhood of individuals on the frontier identified '
                                                 'by DeepJanus')
    parser.add_argument('individuals', metavar='ind', type=int, nargs='+',
                        help='an individual whose neighborhood to explore')
    args = parser.parse_args()

    explore_neighborhood(args.individuals)
