import cv2
import os
from natsort import natsorted

# Path to the Weizmann dataset folder
dataset_path = r"C:\Users\velaga mouli\OneDrive\Desktop\Weizmann"  # Change this to your actual dataset path
output_video_name = "combined_video.avi"  # Final output video file

# Get the list of action folders (run, walk, wave, jump)
action_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

# List to store all images from different folders
all_images = []

# Process each action folder and collect image paths
for action in action_folders:
    action_path = os.path.join(dataset_path, action)
    # Get sorted list of images in the action folder
    images = [img for img in os.listdir(action_path) if img.endswith(".png") or img.endswith(".jpg")]
    images = natsorted(images)  # Sort numerically
    # Store full image paths
    all_images.extend([os.path.join(action_path, img) for img in images])

# Read the first image to get dimensions
first_image = cv2.imread(all_images[0])
height, width, layers = first_image.shape

# Define video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec format
video = cv2.VideoWriter(output_video_name, fourcc, 10, (width, height))  # 10 FPS

# Write each image as a video frame
for img_path in all_images:
    frame = cv2.imread(img_path)
    video.write(frame)

# Release resources
video.release()

print(f"Single combined video saved as {output_video_name}")
