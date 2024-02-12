import numpy as np

from .archive import Archive
from .individual import Individual


def calculate_seed_radius(solution: Archive):
    """Calculates the distance between each member outside the frontier and the seed (mindist metric)"""
    if len(solution) == 0:
        return None
    distances = []
    i: Individual
    for i in solution:
        dist = i.mbr.distance(i.seed)
        distances.append(dist)
    radius = np.mean(distances)
    return radius.item()


def calculate_diameter(solution: Archive):
    """Calculates the distance between each member and the farthest element of the solution (diameter metric)"""
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
