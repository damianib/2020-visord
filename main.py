import time
from main_process import process_video

# CONFIG

source_path = "resources/original-face.mp4"
out_path = "output/output.avi"
downsample_level = 3
lowcut = 50 / 60
highcut = 60 / 60
fs = 30
order = 5
alpha = 1
chrome_attenuation = 1
distance_threshold = 10

# PROCESSING

print("Starting processing video...")
begin_time = time.time()

process_video(source_path, out_path, downsample_level, lowcut, highcut, fs,
              order, alpha, chrome_attenuation, distance_threshold)

end_time = time.time()
print(f"Processing finished. Time taken : {end_time - begin_time}")
