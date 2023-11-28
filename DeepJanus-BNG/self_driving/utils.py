import numpy as np

RoadNodes = list[tuple[float, float, float, float]]
List2DTuple = list[tuple[float, float]]
Point2D = tuple[float, float]
Point4D = tuple[float, float, float, float]


def get_node_coords(node):
    return node[0], node[1], node[2]


def points_distance(p1, p2):
    return np.linalg.norm(np.subtract(get_node_coords(p1), get_node_coords(p2)))
