from typing import Tuple, Any
import numpy as np
import cv2

NumpyArray = Any


def convert_video_to_np_array(file_path: str) -> NumpyArray:
    images = []
    cap = cv2.VideoCapture(file_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            images.append(frame)
        if not ret:
            cap.release()
    all_frames = np.array(images)
    return all_frames


def read_np_array_as_video(frames: NumpyArray, auto_destroy: bool = True) -> None:
    for frame in frames:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    if auto_destroy:
        cv2.destroyAllWindows()


def convert_np_array_to_video(frames: NumpyArray, out_path: str, frames_per_sec: int, codec: str = 'DIVX') -> None:
    width, height = get_width_and_height_from_frames(frames)
    print(width, height)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter('output/output.avi', fourcc, frames_per_sec, (width, height))
    for frame in frames:
        out.write(frame)
    # Release everything if job is finished
    out.release()


def get_width_and_height_from_frames(frames: NumpyArray) -> Tuple[int, int]:
    width, height = frames[0].shape[0], frames[0].shape[1]
    return width, height

frames = convert_video_to_np_array('resources/result-face.mp4')
#read_np_array_as_video(frames)

frames_per_sec = 20
convert_np_array_to_video(frames, 'output/output.avi', frames_per_sec)
