from filters import binomial_filter_1d, binomial_filter_2d
from video_helpers import convert_video_to_np_array, read_np_array_as_video, convert_np_array_to_video

print(binomial_filter_2d(5))

#frames, fps = convert_video_to_np_array('resources/result-face.mp4')
#read_np_array_as_video(frames)
#convert_np_array_to_video(frames, 'output/output.avi', fps)

