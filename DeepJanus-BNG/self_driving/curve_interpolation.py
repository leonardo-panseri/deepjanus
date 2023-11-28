import numpy as np
import matplotlib.pyplot as plt

from self_driving.utils import Point2D, Point4D


def catmull_rom_spline(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D, num_points=20):
    """Calculates interpolation points for the curve between points p1 and p2 using Catmull-Rom Spline formula.
    Parameters p0, p1, p2, and p3 should be (x,y) tuples that define points in the 2D space.
    Parameter num_points is the number of interpolation points to include in this curve segment."""
    # Convert the points to numpy so that we can do array multiplication
    p0, p1, p2, p3 = map(np.array, [p0, p1, p2, p3])

    # Calculate t0 to t4
    # For knot parametrization
    alpha = 0.5

    def tj(ti, p_i, p_j):
        xi, yi = p_i
        xj, yj = p_j
        return (((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5) ** alpha + ti

    # Knot sequence
    t0 = 0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    # Only calculate points between p1 and p2
    t = np.linspace(t1, t2, num_points)

    # Reshape so that we can multiply by the points p0 to p3
    # and get a point for each value of t.
    t = t.reshape(len(t), 1)

    def interp(_t, _ti, _tj, _pu, _pv):
        return (_tj - _t) / (_tj - _ti) * _pu + (_t - _ti) / (_tj - _ti) * _pv

    a1 = interp(t, t0, t1, p0, p1)  # (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = interp(t, t1, t2, p1, p2)  # (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = interp(t, t2, t3, p2, p3)  # (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

    b1 = interp(t, t0, t2, a1, a2)  # (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = interp(t, t1, t3, a2, a3)  # (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3

    c = interp(t, t1, t2, b1, b2)  # (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    return c


def catmull_rom_chain(points: list[Point2D], num_spline_points=20) -> list[Point2D]:
    """Calculates interpolation points for the curve defined by points[1] to points[n-2] using Catmull-Rom Spline.
    Points must be (x,y) tuples.
    Note that the first and last points will not be included in the interpolation, because of Catmull-Rom Spline
    theoretical limitation."""
    if len(points) < 4:
        raise ValueError("points should have at least 4 points")

    # The curve cr will contain an array of (x, y) points.
    cr: list[Point2D] = []
    for i in range(len(points) - 3):
        c = catmull_rom_spline(points[i], points[i + 1], points[i + 2], points[i + 3], num_spline_points)
        if i > 0:
            c = np.delete(c, [0], axis=0)
        cr.extend([tuple(nda) for nda in c])
    return cr


def catmull_rom(points: list[Point4D], num_spline_points=20) -> list[Point4D]:
    """Calculates interpolation points for the curve defined by points[1] to points[n-2] using Catmull-Rom Spline.
    Points must be (x,y,z,w) tuples.
    Note that the first and last points will not be included in the interpolation, because of Catmull-Rom Spline
    theoretical limitation."""
    assert all(x[3] == points[0][3] for x in points)

    xy_points = catmull_rom_chain([(p[0], p[1]) for p in points], num_spline_points)
    z = points[0][2]
    w = points[0][3]
    return [(p[0], p[1], z, w) for p in xy_points]


def plot_catmull_rom(result: list[Point2D], interp_points: list[Point2D]):
    """Plots the results of a curve interpolated with Catmull-Rom Spline."""
    x, y = zip(*result)
    plt.plot(x, y, "bo", markersize=1)
    px, py = zip(*interp_points)
    plt.plot(px, py, 'or', markersize=1)
    plt.show()


if __name__ == '__main__':
    pnts = [(0, 4), (1, 2), (3, 1), (5, 3), (3, 5), (1, 7), (3, 9), (5, 8), (6, 6)]
    cmr = catmull_rom_chain(pnts)
    plot_catmull_rom(cmr, pnts)
