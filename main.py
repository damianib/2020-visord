from typing import Tuple, Any
import numpy as np
import cv2

NumpyArray = Any


def convert_video_to_np_array(file_path: str) -> Tuple[NumpyArray, float]:
    images = []
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            images.append(frame)
        if not ret:
            cap.release()
    all_frames = np.array(images)
    return all_frames, fps


def read_np_array_as_video(frames: NumpyArray, auto_destroy: bool = True) -> None:
    for frame in frames:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    if auto_destroy:
        cv2.destroyAllWindows()


def convert_np_array_to_video(frames: NumpyArray, out_path: str, fps: float, codec: str = 'DIVX') -> None:
    width, height = get_width_and_height_from_frames(frames)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    # Release everything if job is finished
    out.release()


def get_width_and_height_from_frames(frames: NumpyArray) -> Tuple[int, int]:
    width, height = frames[0].shape[1], frames[0].shape[0]
    return width, height


frames, fps = convert_video_to_np_array('resources/result-face.mp4')
read_np_array_as_video(frames)
convert_np_array_to_video(frames, 'output/output.avi', fps)
