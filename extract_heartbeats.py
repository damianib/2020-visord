import numpy as np
from video_helpers import NumpyArray
from collections import Counter
import matplotlib.pyplot as plt


def get_heartbeats(processed_frames: NumpyArray, fps: int) -> float:
    """Return main heartbeat frequency found in processed_frames"""

    # Find the most non-zero pixel and apply ftt on it
    not_zero = processed_frames[:, :, :, 0] != 0
    not_zero_indexes = np.nonzero(not_zero)
    not_zero_indexes = list(zip(not_zero_indexes[1], not_zero_indexes[2]))
    best_pixel = Counter(not_zero_indexes).most_common()[0][0]

    # Clip the beginning of the video with no variations
    first_non_zero_value = np.nonzero(processed_frames[:, best_pixel[0], best_pixel[1], 0] > 10)[0][0]

    # Apply FFT on clipped signal
    fft = np.fft.fft(processed_frames[first_non_zero_value:, best_pixel[0], best_pixel[1], 0])
    freq = np.fft.fftfreq(len(fft))
    abs_fft = abs(fft)

    # Find main frequency of signal
    abs_fft[0] = 0
    max_index = np.argmax(abs_fft)
    best_freq = abs(freq[max_index])

    # Convert to number of heartbeats in a minute
    heartbeats = best_freq * fps * 60

    # Plot signal and fft
    # plt.plot(processed_frames[first_non_zero_value:, best_pixel[0], best_pixel[1], 0])
    # plt.show()
    # plt.plot(freq, abs_fft)
    # plt.show()

    return heartbeats
