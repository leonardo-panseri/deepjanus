import numpy as np

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Point4D = tuple[float, float, float, float]


def get_node_coords(node: Point4D) -> Point3D:
    return node[0], node[1], node[2]


def points_distance(p1, p2):
    return np.linalg.norm(np.subtract(get_node_coords(p1), get_node_coords(p2)))
