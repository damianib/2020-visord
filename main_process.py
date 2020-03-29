import cv2
from project_types import NumpyArray
from spatial_processing import blur_and_downsample_video
from temporal_processing import frequential_filter_video
from video_helpers import convert_video_to_np_array, convert_np_array_to_video, \
    get_width_and_height_from_frames, read_np_array_as_video
import numpy as np


def merge_with_processed(original_frames: NumpyArray, processed_frames: NumpyArray, alpha: float,
                         chrome_attenuation: float) -> NumpyArray:

    dest_width, dest_height = get_width_and_height_from_frames(original_frames)
    images = []
    for index, frame in enumerate(processed_frames):
        # Amplification
        h, s, v = cv2.split(frame)
        h *= alpha
        s *= alpha * chrome_attenuation
        v *= alpha * chrome_attenuation
        amplified_frame = cv2.merge((h, s, v))
        # Merging
        resized_amplified_frame = cv2.resize(amplified_frame, (dest_width, dest_height))
        original_frame = original_frames[index]
        merged = cv2.addWeighted(original_frame, 1, resized_amplified_frame, 0.3, 0)
        images.append(merged)
    return np.array(images).astype(np.uint8)


def process_video(source_path: str, out_path: str, downsample_level: int, lowcut: float, highcut: float, fs: float,
                  order: int, alpha: float, chrome_attenuation: float) -> None:
    frames, fps = convert_video_to_np_array(source_path)
    spatially_processed_frames = blur_and_downsample_video(frames, downsample_level)
    processed_frames = frequential_filter_video(spatially_processed_frames, lowcut, highcut, fs, order)
    merged_frames = merge_with_processed(frames, processed_frames, alpha, chrome_attenuation)
    convert_np_array_to_video(merged_frames, out_path, fps)
