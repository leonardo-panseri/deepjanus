from datetime import datetime
import json
import random
import timeit

import numpy
from deap import base
from deap import creator
from deap import tools

from .evaluator import Evaluator
from .folders import delete_folder_recursively, FolderStorage
from .individual import Individual
from .log import get_logger, log_setup
from .member import Member
from .problem import Problem

log = get_logger(__file__)


def main(problem: Problem, seed:  int | float | str | bytes | bytearray = None, restart_from_last_gen=False,
         log_to_terminal=True, log_to_file=True):
    """
    Executes the DeepJanus algorithm on a given problem.
    :param problem: class representing the search problem that DeepJanus needs to solve
    :param seed: seed to be used for all pseudorandom operations in this run of DeepJanus
    :param restart_from_last_gen: if the experiment should be restarted keeping all the previous generations
    :param log_to_terminal: if logs should be printed to terminal
    :param log_to_file: if logs should be saved to file (the file will be created in the experiment folder)
    :return: a tuple containing the population of the final generation and the logbook where generations
     statistics are saved
    """
    config = problem.config
    random.seed(seed)

    # ####################
    # DEAP framework setup
    # ####################

    # Bi-objective fitness function:
    # 1. Maximize the sparseness among the individuals in the archive
    # 2. Minimize the distance to the frontier of behavior
    creator.create("FitnessMulti", base.Fitness, weights=tuple(config.FITNESS_WEIGHTS))
    # Individuals will be represented by a custom class, based on the problem
    # Their fitness will be evaluated by the bi-objective fitness function defined above
    creator.create("Individual", problem.deap_individual_class(), fitness=creator.FitnessMulti)

    # Save the reference to the individual creator
    problem.individual_creator = creator.Individual

    # Toolbox that will contain all operators needed by DeepJanus evolutionary algorithm
    toolbox = base.Toolbox()
    problem.toolbox = toolbox

    # Method to generate an individual from the seed pool
    toolbox.register("individual", problem.deap_generate_individual)
    # Method to initialize the population as a list of individuals generated with the function above
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Method that assigns the two fitness values to an individual and return them in a tuple
    toolbox.register("evaluate", problem.deap_evaluate_individual)
    # Method that mutates an individual
    toolbox.register("mutate", problem.deap_mutate_individual)
    # Standard NSGA2 selector that chooses which individuals to keep based on the
    # smallest non-domination rank and the highest crowding distance
    toolbox.register("select", tools.selNSGA2)
    # Method to generate the offspring of the previous generation using a tournament
    # with the NSGA2 selection strategy explained above
    toolbox.register("offspring", tools.selTournamentDCD)
    # Method to update archive based on a new population
    toolbox.register("update_archive", problem.deap_update_archive)
    # Method to substitute individuals that are evolved from a seed that already generated a solution in the archive
    # with new individuals generated from another seed that has not generated a solution in the archive
    toolbox.register("repopulate", problem.deap_repopulate)

    # Module to collect statistics for the fitness values of all individuals
    stats = tools.Statistics(lambda i: i.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"

    # ####################
    # DeepJanus algorithm
    # ####################

    if not restart_from_last_gen:
        delete_folder_recursively(problem.experiment_path)
        problem.experiment_path.mkdir(parents=True, exist_ok=True)

    if log_to_terminal:
        log_setup.setup_console_log(problem.config.FOLDERS.log_ini)
    if log_to_file:
        log_setup.setup_file_log(problem.experiment_path
                                 .joinpath(datetime.strftime(datetime.now(), '%d-%m-%Y_%H-%M-%S') + '.log'))

    exp_start = timeit.default_timer()

    if restart_from_last_gen:
        start_gen, pop = load_last_gen_data(problem)
        # This will only assign the crowding distance to the individuals (no actual selection is done)
        pop = toolbox.select(pop, config.POP_SIZE)
        log.info(f'### Restarting from generation {start_gen} '
                 f'- loaded {len(pop)} individuals ({len(problem.archive)} in archive)')
    else:
        start_gen = 0

        # Customizable callback to execute actions at the start of the experiment
        problem.on_experiment_start()

        # Generate initial population
        log.info("### Initializing population...")
        pop = toolbox.population(n=config.POP_SIZE)

    # Begin the generational process
    for gen in range(start_gen, config.NUM_GENERATIONS):
        log.info(f"### Generation {gen}")
        gen_start = timeit.default_timer()

        # Customizable callback to execute actions at the start of each iteration
        problem.on_iteration_start(gen)

        offspring = []
        if gen == 0:
            # Evaluate the initial population
            individuals_to_eval = pop
        else:
            # Generate the offspring of the previous generation
            offspring = [ind.clone(problem.individual_creator) for ind in toolbox.offspring(pop, len(pop))]

            # Mutate offspring
            for ind in offspring:
                toolbox.mutate(ind)
                del ind.fitness.values

            # Repopulate by substituting descendants of seeds that already generated a solution
            new_individuals = toolbox.repopulate(pop)

            # Choose the individuals to evaluate: all in offspring because they are mutated and any individual
            # substituted by re-population
            individuals_to_eval = offspring + new_individuals

        # Evaluate the individuals
        fitness = toolbox.map(toolbox.evaluate, individuals_to_eval)
        # TODO: should archive sparseness be normalized?
        for ind, fit in zip(individuals_to_eval, fitness):
            ind.fitness.values = fit

        # Update the archive
        toolbox.update_archive(individuals_to_eval)

        # Select the next generation population
        # For generation 0, this will only assign the crowding distance to the individuals (no actual selection is done)
        pop = toolbox.select(pop + offspring, config.POP_SIZE)

        # Save all selected individuals of this generation to disk
        for ind in pop:
            ind.save(problem.current_population_path, False)

        # Calculate statistics for the current generation
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(individuals_to_eval), **record)
        log.info(f"Generation {gen} stats:\n{logbook.header}\n{logbook.stream}")

        # Customizable callback to execute actions at the end of each iteration
        problem.on_iteration_end(gen, record)

        hours, remainder = divmod(timeit.default_timer() - gen_start, 3600)
        minutes, seconds = divmod(remainder, 60)
        log.info(f"Time for generation {gen}: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    # Customizable callback to execute actions at the end of the experiment
    problem.on_experiment_end(logbook)

    hours, remainder = divmod(timeit.default_timer() - exp_start, 3600)
    minutes, seconds = divmod(remainder, 60)
    log.info(f"Time for experiment: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    return pop, logbook


def load_last_gen_data(problem: Problem):
    status = json.loads(problem.experiment_path.joinpath('status.json').read_text())

    ind_counter = status['ind_counter']
    Individual.counter = ind_counter
    Member.counter = ind_counter

    last_gen_idx = status['last_gen']
    delete_folder_recursively(problem.experiment_path.joinpath(f'gen{last_gen_idx + 1}'))

    pop = []
    pop_storage = FolderStorage(problem.experiment_path.joinpath(f'gen{last_gen_idx}', 'population'), 'ind{}.json')
    for path in pop_storage.all_files('*.json'):
        ind: Individual = Individual.from_dict(pop_storage.load_json_by_path(path), problem.individual_creator,
                                               problem.member_class())
        lb, ub = ind.unsafe_region_probability
        ind.fitness.values = (Evaluator.calculate_fitness_functions(ind.sparseness,
                                                                    problem.config.PROBABILITY_THRESHOLD, lb, ub))
        pop.append(ind)

    archive_storage = FolderStorage(problem.experiment_path.joinpath(f'gen{last_gen_idx}', 'archive'), 'ind{}.json')
    for path in archive_storage.all_files('*.json'):
        ind: Individual = Individual.from_dict(archive_storage.load_json_by_path(path), problem.individual_creator,
                                               problem.member_class())
        lb, ub = ind.unsafe_region_probability
        ind.fitness.values = (Evaluator.calculate_fitness_functions(ind.sparseness,
                                                                    problem.config.PROBABILITY_THRESHOLD, lb, ub))
        problem.archive.add(ind)

    return last_gen_idx + 1, pop
