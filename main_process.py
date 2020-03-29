import cv2
from project_types import NumpyArray
from spatial_processing import blur_and_downsample_video
from temporal_processing import frequential_filter_video
from video_helpers import convert_video_to_np_array, convert_np_array_to_video, \
    get_width_and_height_from_frames, read_np_array_as_video
import numpy as np


def merge_with_processed(original_frames: NumpyArray, processed_frames: NumpyArray, alpha: float,
                         chrome_attenuation: float) -> NumpyArray:
    # Amplification
    reduced_width, reduced_height = get_width_and_height_from_frames(processed_frames)
    dest_width, dest_height = get_width_and_height_from_frames(original_frames)
    total_time = processed_frames.shape[0]
    processed_frames_amplified = np.zeros((total_time, reduced_height, reduced_width, 3))
    processed_frames_amplified[:][:][:][0] = processed_frames[:][:][:][0] * alpha
    processed_frames_amplified[:][:][:][1] = processed_frames[:][:][:][0] * alpha * chrome_attenuation
    processed_frames_amplified[:][:][:][2] = processed_frames[:][:][:][0] * alpha * chrome_attenuation
    # Merging
    merged = original_frames + cv2.resize(processed_frames_amplified, (dest_height, dest_width))
    return merged


def process_video(source_path: str, out_path: str, downsample_level: int, lowcut: float, highcut: float, fs: float,
                  order: int, alpha: float, chrome_attenuation: float) -> None:
    frames, fps = convert_video_to_np_array(source_path)
    spatially_processed_frames = blur_and_downsample_video(frames, downsample_level)
    processed_frames = frequential_filter_video(spatially_processed_frames, lowcut, highcut, fs, order)
    print(frames.shape)
    print(processed_frames.shape)
    print(processed_frames)
    convert_np_array_to_video(processed_frames, out_path, fps)
    #merged_frames = merge_with_processed(frames, processed_frames, alpha, chrome_attenuation)
    #convert_np_array_to_video(merged_frames, out_path, fps)
