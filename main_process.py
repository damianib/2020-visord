import cv2
from project_types import NumpyArray
from spatial_processing import blur_and_downsample_video
from temporal_processing import frequency_filter_video
from video_helpers import convert_video_to_np_array, convert_np_array_to_video, \
    get_width_and_height_from_frames
import numpy as np


def h_distance(h1: int, h2: int):
    """Distance between two hue in hsv format"""
    return min(abs(h1 - h2), 360 - abs(h1 - h2))


def merge_with_processed(original_frames: NumpyArray, processed_frames: NumpyArray, alpha: float,
                         chrome_attenuation: float, distance_threshold: int, downsample_level: int) -> NumpyArray:
    """Merge the original frames of the video with the processed frames (spatially and temporaly).
    Alpha and chrome attenuation are used to prettify the merge of the 2 array of frames.
    Distance threshold is used to only keep colors close to red."""
    dest_width, dest_height = get_width_and_height_from_frames(original_frames)
    images = []
    for index, frame in enumerate(processed_frames):
        # Amplification
        h, s, v = cv2.split(frame)
        h *= alpha
        s *= alpha * chrome_attenuation
        v *= alpha * chrome_attenuation
        amplified_frame = cv2.merge((h, s, v))
        # Getting rid of colors other than red (h=0)
        h, s, v = cv2.split(amplified_frame)
        distances = np.array([[h_distance(h[i, j], 0) for j in range(h.shape[1])] for i in range(h.shape[0])])
        h *= (distances < distance_threshold)
        s *= (distances < distance_threshold)
        v *= (distances < distance_threshold)
        filtered_frame = cv2.merge((h, s, v))
        # Resizing
        for i in range(downsample_level):
            filtered_frame = cv2.pyrUp(filtered_frame)
        resized_filtered_frame = cv2.resize(filtered_frame, (dest_width, dest_height))
        # Merging
        original_frame = original_frames[index]
        merged = cv2.addWeighted(original_frame, 0.8, resized_filtered_frame, 0.2, 0)
        images.append(merged)
    return np.array(images).astype(np.uint8)


def process_video(source_path: str, out_path: str, downsample_level: int, lowcut: float, highcut: float, fs: float,
                  order: int, alpha: float, chrome_attenuation: float, distance_threshold: int) -> None:
    """Main function which processes the video located at the source path provided and output the result to the out
    path provided.
    Downsample level is used for the spatial processing of the frames.
    lowcut, highcut, fs and order are used to build the bandpass filter for the temporal processing.
    alpha and chrome attenuation are used to prettify the merge of the original frames and the processed frames."""
    frames, fps = convert_video_to_np_array(source_path)
    spatially_processed_frames = blur_and_downsample_video(frames, downsample_level)
    processed_frames = frequency_filter_video(spatially_processed_frames, lowcut, highcut, fs, order)
    merged_frames = merge_with_processed(frames, processed_frames, alpha, chrome_attenuation, distance_threshold,
                                         downsample_level)
    convert_np_array_to_video(merged_frames, out_path, fps)
