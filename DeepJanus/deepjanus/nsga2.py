import random
import timeit

import numpy
from deap import base
from deap import creator
from deap import tools

from .log import get_logger
from .problem import Problem

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

    exp_start = timeit.default_timer()

    # Customizable callback to execute actions at the start of the experiment
    problem.on_experiment_start()

    # Generate initial population
    log.info("### Initializing population...")
    pop = toolbox.population(n=config.POP_SIZE)

    # Begin the generational process
    for gen in range(config.NUM_GENERATIONS):
        log.info(f"### Generation {gen}")
        gen_start = timeit.default_timer()

        # Customizable callback to execute actions at the start of each iteration
        problem.on_iteration_start(gen)

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
        for ind, fit in zip(individuals_to_eval, fitness):
            ind.fitness.values = fit

        # Update the archive
        toolbox.update_archive(individuals_to_eval)

        # Select the next generation population
        # For generation 0, this will only assign the crowding distance to the individuals (no actual selection is done)
        pop = toolbox.select(individuals_to_eval, config.POP_SIZE)

        # Calculate statistics for the current generation
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(individuals_to_eval), **record)
        log.info(f"Generation {gen} stats:\n{logbook.stream}")

        # Customizable callback to execute actions at the end of each iteration
        problem.on_iteration_end(record)

        hours, remainder = divmod(timeit.default_timer() - gen_start, 3600)
        minutes, seconds = divmod(remainder, 60)
        log.info(f"Time for generation {gen}: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    # Customizable callback to execute actions at the end of the experiment
    problem.on_experiment_end(logbook)

    hours, remainder = divmod(timeit.default_timer() - exp_start, 3600)
    minutes, seconds = divmod(remainder, 60)
    log.info(f"Time for experiment: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    return pop, logbook


if __name__ == "__main__":
    final_population, search_stats = main()
