import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Paths
dataset_path = r"C:\Users\velaga mouli\OneDrive\Desktop\Weizmann"  # Update with your dataset path
classes = ["walk", "run", "jump", "wave"]  # Class labels

# Parameters
image_size = (128, 128)  # Resize all images to 128x128
data = []
labels = []

# Load images and labels
for label, action in enumerate(classes):
    action_path = os.path.join(dataset_path, action)
    for file in os.listdir(action_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(action_path, file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            data.append(image)
            labels.append(label)

# Convert to numpy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values (0-1)
labels = np.array(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("action_recognition_model.h5")
print("Model saved successfully!")
