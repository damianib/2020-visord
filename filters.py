import numpy as np

from project_types import NumpyArray


def binomial_filter_1d(size: int = 2) -> NumpyArray:
    """ A binomial 1D filter, corresponding to a row of the Pascal Triangle.
    It can be obtained by convoluting [1, 1] with itself as many time as needed, because of the Newton's Formula for
    (a+b)**n.
    To use a set of these coefficients as a low pass filter, the values must be normalized so the sum is one."""
    return (np.poly1d([0.5, 0.5]) ** (size - 1)).coeffs


def binomial_filter_2d(size: int = 2) -> NumpyArray:
    """ A binomial 2D filter, obtained by multiplying 2 binomial 1D filters."""
    filter_1d = binomial_filter_1d(size)
    filter_2d = np.outer(filter_1d, filter_1d)
    return filter_2d