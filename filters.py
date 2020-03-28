import cv2
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


def blur_and_downsample_all_channels(img: NumpyArray, filt: NumpyArray, level: int = 1) -> NumpyArray:
    """ Blur and downsample all three channels of a HSV or RGB image, represented by a Numpy Matrix.
          The blurring is done with the 2D filter kernel specified by filt, which has to be a Numpy Matrix.
          The procedure will be applied recursively level times (default=1)."""
    comp1, comp2, comp3 = cv2.split(img)
    processed_comp1 = blur_and_downsample_one_channel(comp1, filt, level)
    processed_comp2 = blur_and_downsample_one_channel(comp2, filt, level)
    processed_comp3 = blur_and_downsample_one_channel(comp3, filt, level)
    result = cv2.merge((processed_comp1, processed_comp2, processed_comp3))
    return result


def blur_and_downsample_one_channel(img: NumpyArray, filt: NumpyArray, level: int = 1) -> NumpyArray:
    """ Blur and downsample one channel of image, represented by a Numpy Matrix.
      The blurring is done with the 2D filter kernel specified by filt, which has to be a Numpy Matrix.
      The procedure is applied recursively level times (default=1)."""
    if level > 1:
        img = blur_and_downsample_one_channel(img, level - 1, filt)

    if level >= 1:
        res = corrDn(img, filter, step=(2, 2))
    else:
        res = img

    return res
