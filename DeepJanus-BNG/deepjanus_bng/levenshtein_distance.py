from typing import List, Tuple
import numpy as np

from .points import Point4D, Point2D

AngleLength = Tuple[float, float]
ListOfAngleLength = List[AngleLength]


def _calc_cost_discrete(u: AngleLength, v: AngleLength) -> float:
    delta_angle, delta_len = np.subtract(u, v)
    delta_angle = np.abs((delta_angle + 180) % 360 - 180)
    eps_angle = 0.3
    eps_len = 0.2
    if delta_angle < eps_angle and delta_len < eps_len:
        res = 0
    else:
        res = 2
    return res


def _calc_cost_weighted(u: AngleLength, v: AngleLength) -> float:
    delta_angle, delta_len = np.abs(np.subtract(u, v))
    delta_angle = np.abs((delta_angle + 180) % 360 - 180)
    eps_angle = 0.3
    eps_len = 0.2
    if delta_angle < eps_angle and delta_len < eps_len:
        res = 0
    else:
        res = 1 / 2 * (delta_angle / (1 + delta_angle) + delta_len / (1 + delta_len))
    return res


def _iterative_levenshtein_dist_angle(s: ListOfAngleLength, t: ListOfAngleLength, cost_function=_calc_cost_weighted):
    """ Calculates the Levenshtein distance between two objects s and t. For all i and j, dist[i,j] will contain
    the Levenshtein distance between the first i elements of s and the first j elements of t."""
    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[.0 for _ in range(cols)] for _ in range(rows)]

    # Source prefixes can be transformed into empty objects by deletions
    for i in range(1, rows):
        dist[i][0] = i

    # Target prefixes can be created from an empty source object by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            cost = cost_function(s[row - 1], t[col - 1])
            dist[row][col] = min(dist[row - 1][col] + 1.0,  # deletion
                                 dist[row][col - 1] + 1.0,  # insertion
                                 dist[row - 1][col - 1] + cost)  # substitution

    return dist[rows-1][cols-1]


def _calc_dist_angle(points: list[Point4D | Point2D]) -> ListOfAngleLength:
    """Prepares a list of points to be processed with Levenshtein distance method."""
    assert len(points) >= 2, f'at least two points are needed'

    def vector(idx) -> np.ndarray:
        return np.subtract(points[idx + 1], points[idx])

    def find_angle(v0: np.ndarray, v1: np.ndarray) -> float:
        at_0 = np.arctan2(v0[1], v0[0])
        at_1 = np.arctan2(v1[1], v1[0])
        return at_1 - at_0

    n = len(points) - 1
    result: ListOfAngleLength = [(0, 0)] * n
    b = vector(0)
    for i in range(n):
        a = b
        b = vector(i)
        angle = np.degrees(find_angle(a, b))
        distance = np.linalg.norm(b)
        result[i] = (angle, distance)
    return result


def iterative_levenshtein(s: list[Point4D | Point2D], t: list[Point4D | Point2D]):
    """Calculates the Levenshtein distance between two set of points."""
    s_da = _calc_dist_angle(s)
    t_da = _calc_dist_angle(t)
    return _iterative_levenshtein_dist_angle(s_da, t_da)


if __name__ == '__main__':
    import unittest

    class TestDist(unittest.TestCase):

        def setUp(self):
            self.s = [(.0, .0), (.0, 2.), (2., 2.)]
            self.t = [(.0, .0), (.0, 2.), (-2., 2.)]

        def test_dist_angle_calculations(self):
            self.assertEqual(_calc_dist_angle(self.s), [(0, 2), (-90, 2)])
            self.assertEqual(_calc_dist_angle(self.t), [(0, 2), (90, 2)])

        def test_iterative_levenshtein_dist_angle(self):
            s = [(2, -90), (2, 0)]
            t = [(2, -90), (2, 0)]
            dist = _iterative_levenshtein_dist_angle(s, t)
            self.assertEqual(dist, 0)

        def test_iterative_levenshtein(self):
            dist = iterative_levenshtein(self.t, self.s)
            self.assertNotEqual(dist, 0)

    unittest.main()
