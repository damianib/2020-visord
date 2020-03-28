import cv2
import numpy as np
from project_types import NumpyArray


def blur_and_downsample(img: NumpyArray, level: int = 2) -> NumpyArray:
    img_copy = img.copy()
    for i in range(level - 1):
        img_copy = cv2.pyrDown(img_copy)
    return img_copy


def blur_and_downsample_video(frames: NumpyArray, level: int = 2) -> NumpyArray:
    images = []
    for frame in frames:
        images.append(blur_and_downsample(frame, level))
    all_frames = np.array(images)
    return all_frames
