import os
import json
import random
import shutil

# Set the paths
dataset_path = r"C:\Users\msair\Downloads\CViT-main\CViT-main\dfdc_train_part_0"
subset_path_70 = r"C:\Users\msair\Downloads\CViT-main\CViT-main\preprocessing\deal\dfdc_train_part_3"
subset_path_20 = r"C:\Users\msair\Downloads\CViT-main\CViT-main\preprocessing\deal\dfdc_train_part_36"
subset_path_10 = r"C:\Users\msair\Downloads\CViT-main\CViT-main\preprocessing\deal\dfdc_train_part_48"
metadata_file = os.path.join(dataset_path, "metadata.json")

# Load the metadata
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Get a list of real videos and their corresponding fake videos
real_videos = []
fake_videos = []
for filename, video_info in metadata.items():
    if video_info["label"] == "REAL":
        real_videos.append(filename)
    elif video_info["label"] == "FAKE":
        fake_videos.append((filename, video_info["original"]))

# Limit the total number of real videos to 200
real_videos = random.sample(real_videos, min(400, len(real_videos)))

# Get the corresponding fake videos for the selected real videos
fake_videos = [(fake, original) for fake, original in fake_videos if original in real_videos]

# Randomly select videos for each subset
random.shuffle(real_videos)
random.shuffle(fake_videos)

num_real_videos_70 = int(0.7 * len(real_videos))
num_real_videos_20 = int(0.2 * len(real_videos))
num_real_videos_10 = len(real_videos) - num_real_videos_70 - num_real_videos_20

subset_real_videos_70 = real_videos[:num_real_videos_70]
subset_real_videos_20 = real_videos[num_real_videos_70:num_real_videos_70+num_real_videos_20]
subset_real_videos_10 = real_videos[num_real_videos_70+num_real_videos_20:]

subset_fake_videos_70 = [fake for fake in fake_videos if fake[1] in subset_real_videos_70]
subset_fake_videos_20 = [fake for fake in fake_videos if fake[1] in subset_real_videos_20]
subset_fake_videos_10 = [fake for fake in fake_videos if fake[1] in subset_real_videos_10]

# Create the subset directories
os.makedirs(subset_path_70, exist_ok=True)
os.makedirs(subset_path_20, exist_ok=True)
os.makedirs(subset_path_10, exist_ok=True)

# Copy the selected videos to the subset directories
for video in subset_real_videos_70:
    src = os.path.join(dataset_path, video)
    dst = os.path.join(subset_path_70, video)
    shutil.copy(src, dst)

for video in subset_real_videos_20:
    src = os.path.join(dataset_path, video)
    dst = os.path.join(subset_path_20, video)
    shutil.copy(src, dst)

for video in subset_real_videos_10:
    src = os.path.join(dataset_path, video)
    dst = os.path.join(subset_path_10, video)
    shutil.copy(src, dst)

for fake_video, real_video in subset_fake_videos_70:
    src = os.path.join(dataset_path, fake_video)
    dst = os.path.join(subset_path_70, fake_video)
    shutil.copy(src, dst)

for fake_video, real_video in subset_fake_videos_20:
    src = os.path.join(dataset_path, fake_video)
    dst = os.path.join(subset_path_20, fake_video)
    shutil.copy(src, dst)

for fake_video, real_video in subset_fake_videos_10:
    src = os.path.join(dataset_path, fake_video)
    dst = os.path.join(subset_path_10, fake_video)
    shutil.copy(src, dst)

# Copy the metadata file to the subset directories
shutil.copy(metadata_file, subset_path_70)
shutil.copy(metadata_file, subset_path_20)
shutil.copy(metadata_file, subset_path_10)
