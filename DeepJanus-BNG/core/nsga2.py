import random

import numpy
from deap import base
from deap import creator
from deap import tools

from core.log import get_logger
from core.problem import Problem

log = get_logger(__file__)


def main(problem: Problem = None, seed:  int | float | str | bytes | bytearray = None):
    """
    Executes the DeepJanus algorithm on a given problem.
    :param problem: class representing the search problem that DeepJanus needs to solve
    :param seed: seed to be used for all pseudorandom operations in this run of DeepJanus
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
    creator.create("FitnessMulti", base.Fitness, weights=config.FITNESS_WEIGHTS)
    # Individuals will be represented by a custom class, based on the problem
    # Their fitness will be evaluated by the bi-objective fitness function defined above
    creator.create("Individual", problem.deap_individual_class(), fitness=creator.FitnessMulti)

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
    toolbox.register("update_archive", problem.archive.process_population)
    # Method to substitute individuals that are evolved from a seed that already generated a solution in the archive
    # with new individuals generated from another seed that has not generated a solution in the archive
    toolbox.register("repopulate", problem.reseed)

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

    # Generate initial population
    log.info("### Initializing population....")
    pop = toolbox.population(n=config.POP_SIZE)

    # Evaluate the initial population
    # Note: the fitness functions are all invalid before the first iteration since they have not been evaluated.
    # individuals_to_eval = [ind for ind in pop if not ind.fitness.valid]
    # TODO check if this is needed: problem.pre_evaluate_members(individuals_to_eval)
    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    # Initialize the archive
    toolbox.update_archive(pop)

    # This is just to assign the crowding distance to the individuals (no actual selection is done)
    pop = toolbox.select(pop, len(pop))

    # Calculate statistics for the initial generation
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    log.info(f"Generation {0} stats:\n{logbook.stream}")

    # Problem-specific callback to execute actions at each iteration (e.g., save data to file)
    problem.on_iteration(0, pop, logbook)

    # Begin the generational process
    for gen in range(1, config.NUM_GENERATIONS):
        # Generate the offspring of the previous generation
        offspring = [ind.clone() for ind in toolbox.offspring(pop, len(pop))]

        # Mutate offspring
        for ind in offspring:
            toolbox.mutate(ind)
            del ind.fitness.values

        # Repopulate by substituting descendants of seeds that already generated a solution
        toolbox.repopulate(pop, offspring)

        # Evaluate the individuals
        individuals_to_eval = offspring + pop
        # TODO check if this is needed: problem.pre_evaluate_members(individuals_to_eval)
        fitness = toolbox.map(toolbox.evaluate, individuals_to_eval)
        for ind, fit in zip(individuals_to_eval, fitness):
            ind.fitness.values = fit

        # Update the archive
        toolbox.update_archive(offspring + pop)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, config.POP_SIZE)

        # Calculate statistics for the current generation
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(individuals_to_eval), **record)
        log.info(f"Generation {gen} stats:\n{logbook.stream}")

        # Problem-specific callback to execute actions at each iteration (e.g., save data to file)
        problem.on_iteration(gen, pop, logbook)

    return pop, logbook


if __name__ == "__main__":
    final_population, search_stats = main()
