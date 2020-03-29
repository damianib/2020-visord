from typing import Tuple
import numpy as np
import cv2
from project_types import NumpyArray

""" More informations about how to handle video with OpenCV here :
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html"""


def convert_video_to_np_array(file_path: str) -> Tuple[NumpyArray, float]:
    """ Read the video file located at the provided file path and, using opencv, convert it to a numpy array.
    Each element of the  array represents a frame of the video. Each frame of the video is itself a matrix of pixel
    (a numpy array of the rows of the image). Each pixel is a numpy array with a length of 3, representing HSV colors.
    As it ease the other operations, we automatically does the BGR to HSV conversion here."""
    images = []
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            images.append(converted_frame)
        if not ret:
            cap.release()
    all_frames = np.array(images)
    return all_frames, fps


def read_np_array_as_video(frames: NumpyArray, ms_per_frame: int = 1, auto_destroy: bool = True,
                           auto_convert: bool = False) -> None:
    """Print the frames of the provided numpy array (HSV representation), one at a time,
    waiting the provided time between each frame.
     If auto destroy is set to True, the plot window is automatically destroyed at the end of the function call."""
    for frame in frames:
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR) if auto_convert else frame
        cv2.imshow('frame', converted_frame)
        cv2.waitKey(ms_per_frame)
    if auto_destroy:
        cv2.destroyAllWindows()


def convert_np_array_to_video(frames: NumpyArray, out_path: str, fps: float, codec: str = 'DIVX') -> None:
    """Given a numpy array representing all the frames of a video in HSV, the out path and the number of frame per seconds,
     save the frames as a video to the provided output path. You can also change the Codec use for this operation.
     The frames are automatically converted from HSV to BGR.
     WARNING : THE CODEC YOU SHOULD USE MAY DEPEND ON YOUR OPERATING SYSTEM"""
    width, height = get_width_and_height_from_frames(frames)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for frame in frames:
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        out.write(converted_frame)
    # Release everything if job is finished
    out.release()


def get_width_and_height_from_frames(frames: NumpyArray) -> Tuple[int, int]:
    """Given a numpy array representing all the frames of a video, returns the width and height of the video.
    Be careful : the dimensions used with numpy are usually ROWS x COLUMNS whereas with video reading/writing,
    the dimensions usually used are WIDTH x HEIGHT (the opposite !)."""
    width, height = frames[0].shape[1], frames[0].shape[0]
    return width, height
