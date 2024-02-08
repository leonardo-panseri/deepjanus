import numpy as np

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Point4D = tuple[float, float, float, float]


def to_3d_point(node: Point4D) -> Point3D:
    """Converts a 4D point to a 3D point."""
    return node[0], node[1], node[2]


def points_distance(p1: Point4D, p2: Point4D):
    """Calculates the distance between two 4D points, ignoring the fourth dimension."""
    return np.linalg.norm(np.subtract(to_3d_point(p1), to_3d_point(p2)))
