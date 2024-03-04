import numpy as np
from typing import TYPE_CHECKING

from .archive import Archive
from .individual import Individual

if TYPE_CHECKING:
    from .problem import Problem


def calculate_seed_radius(problem: 'Problem'):
    """Calculates the distance between each member outside the frontier and the seed (mindist metric)"""
    solution: Archive = problem.archive
    if len(solution) == 0:
        return None
    distances = []
    i: Individual
    for i in solution:
        dist = i.mbr.distance(problem.seed_pool[i.seed_index])
        distances.append(dist)
    radius = np.mean(distances)
    return radius.item()


def calculate_diameter(problem: 'Problem'):
    """Calculates the distance between each member and the farthest element of the solution (diameter metric)"""
    solution: Archive = problem.archive
    if len(solution) == 0:
        return None
    max_distances = []
    for i1 in solution:
        max_dist = .0
        for i2 in solution:
            if i1 != i2:
                dist = i1.distance(i2)
                if dist > max_dist:
                    max_dist = dist
        max_distances.append(max_dist)
    diameter = np.mean(max_distances)
    return diameter.item()
