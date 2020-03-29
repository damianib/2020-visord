import numpy as np
from project_types import NumpyArray
from scipy.signal import butter, sosfilt
from video_helpers import get_width_and_height_from_frames

"""https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html"""


def butter_bandpass(lowcut: float, highcut: float, fs: float, order=5) -> NumpyArray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data: NumpyArray, lowcut: float, highcut: float, fs: float, order=5) -> NumpyArray:
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def frequential_filter_video(spatially_processed_frames: NumpyArray, lowcut: float, highcut: float, fs: float,
                             order=5) -> NumpyArray:
    width, height = get_width_and_height_from_frames(spatially_processed_frames)
    total_time = spatially_processed_frames.shape[0]
    pre_processed_video = np.zeros((3, height, width, total_time))
    post_processed_video = np.zeros((total_time, height, width, 3), dtype=np.uint8)
    # Shape modification to ease frequential filtering
    for t in range(total_time):
        for i in range(height):
            for j in range(width):
                for color in range(3):
                    pre_processed_video[color][i][j][t] = spatially_processed_frames[t][i][j][color]
    # Filtering
    for color in range(3):
        for i in range(height):
            for j in range(width):
                data = pre_processed_video[color][i][j]
                pre_processed_video[color][i][j] = butter_bandpass_filter(data, lowcut, highcut, fs, order=order)
    # Shape modification to return to the original shape
    for color in range(3):
        for i in range(height):
            for j in range(width):
                for t in range(total_time):
                    post_processed_video[t][i][j][color] = int(pre_processed_video[color][i][j][t])
    return post_processed_video
