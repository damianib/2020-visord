from spatial_processing import blur_and_downsample_video
from video_helpers import convert_video_to_np_array, read_np_array_as_video, convert_np_array_to_video


frames, fps = convert_video_to_np_array('resources/result-face.mp4')
#read_np_array_as_video(frames)
spatially_processed_frames = blur_and_downsample_video(frames, 5)
convert_np_array_to_video(spatially_processed_frames, 'output/output.avi', fps)

