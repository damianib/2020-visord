import numpy as np
from project_types import NumpyArray
from scipy.signal import butter, sosfilt
from video_helpers import get_width_and_height_from_frames

"""More informations about using scipy to create a bandpass filter can be found using the links below :
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter"""


def butter_bandpass(lowcut: float, highcut: float, fs: float, order=5) -> NumpyArray:
    """Create a bandpass filter using scipy. The filter filters everything that's not in [lowcut, highcut] and use the
    sample frequency fs provided."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data: NumpyArray, lowcut: float, highcut: float, fs: float, order=5) -> NumpyArray:
    """Given an array of a specific data over time, filters everything that's not in [lowcut, highcut], using the
    sample frequency fs provided."""
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def frequency_filter_video(spatially_processed_frames: NumpyArray, lowcut: float, highcut: float, fs: float,
                           order=5) -> NumpyArray:
    """Convert the array of frames, of shape (total_time, height, width, 3), to a shape (3, height, width, total_time)
    that can be processed by a bandpass filter. The bandpass filter is then apply to every pixel of the image.
    Then the array is converted back to its original shape and returned."""
    width, height = get_width_and_height_from_frames(spatially_processed_frames)
    total_time = spatially_processed_frames.shape[0]
    pre_processed_video = np.zeros((3, height, width, total_time))
    post_processed_video = np.zeros((total_time, height, width, 3), dtype=np.uint8)
    # Shape modification to ease frequency filtering
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
