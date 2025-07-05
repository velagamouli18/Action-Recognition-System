import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load your trained model
model_path = "action_recognition_model.h5"
model = load_model(model_path)

# Actions corresponding to the model's output
actions = ["walk", "run", "jump", "wave"]

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def draw_bounding_box(frame, pose_landmarks, padding=11):
    """Draws a slightly enlarged green bounding box around the detected pose."""
    if pose_landmarks is None:  # Check if pose landmarks exist
        return
    h, w, _ = frame.shape  # Get frame dimensions

    # Compute bounding box based on pose landmarks
    xmin = int(min([lm.x for lm in pose_landmarks.landmark]) * w) - padding
    ymin = int(min([lm.y for lm in pose_landmarks.landmark]) * h) - padding
    xmax = int(max([lm.x for lm in pose_landmarks.landmark]) * w) + padding
    ymax = int(max([lm.y for lm in pose_landmarks.landmark]) * h) + padding

    # Ensure the bounding box stays within the frame
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    # Draw the slightly larger green bounding box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Load the video file
video_path = "combined_video.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the codec and create VideoWriter object
output_path = r"C:\Users\velaga mouli\OneDrive\Desktop\output_combined_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec format
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS from input video
frame_size = (1366, 768)  # Output frame size
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# MediaPipe Pose processing
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to match the model's input size (e.g., 128x128)
        frame_resized = cv2.resize(frame, (128, 128))

        # Normalize the frame and expand dimensions for model input
        frame_expanded = np.expand_dims(frame_resized, axis=0) / 255.0

        # Predict the action for the current frame
        prediction = model.predict(frame_expanded)
        predicted_class = np.argmax(prediction)
        predicted_action = actions[predicted_class]

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw bounding box around the detected pose
        draw_bounding_box(frame, results.pose_landmarks)

        # Display the predicted action label
        cv2.putText(frame, f"Action: {predicted_action}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Resize frame for display and output
        resized_frame = cv2.resize(frame, frame_size)

        # Write frame to output video
        out.write(resized_frame)


# Release resources
cap.release()
out.release()
