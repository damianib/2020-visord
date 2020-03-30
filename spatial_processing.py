import cv2
import numpy as np
from project_types import NumpyArray


def blur_and_downsample(img: NumpyArray, level: int = 1) -> NumpyArray:
    """
    This function takes the image provided and returns the indicated level of its Gaussian Pyramid, using OpenCV.
    Internally, the image is blurred using a binomial filter and then downsampled : 1 row is kept out of 2,
    same for colums.This is down recursively to reach the indicated level.
    More information about the Gaussian Pyramid here :
    https://docs.opencv.org/2.4/doc/tutorials/imgproc/pyramids/pyramids.html
    More informations about its implementation in OpenCV :
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
    """
    img_copy = img.copy()
    for i in range(level):
        img_copy = cv2.pyrDown(img_copy)
    return img_copy


def blur_and_downsample_video(frames: NumpyArray, level: int = 1) -> NumpyArray:
    """
    Apply the blur_and_downsample function to every frame of the provided video and returns the result.
    """
    images = []
    for frame in frames:
        images.append(blur_and_downsample(frame, level))
    all_frames = np.array(images)
    return all_frames
