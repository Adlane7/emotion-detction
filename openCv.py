import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the model structure from a JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights("model_weights.h5")

# Define emotion labels (customize these based on your model)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load the OpenCV face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)  # Assuming the default camera is used

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]

        # Preprocess the face image to match your model's input format
        # For example, resize it to 48x48 pixels and normalize pixel values
        face_img = cv2.resize(face_img, (48, 48))  # Adjust the size as needed
        face_img = face_img / 255.0  # Normalize pixel values to be in the range [0, 1]
        face_img = np.expand_dims(face_img, axis=2)  # Add a channel dimension
        face_img = np.expand_dims(face_img, axis=0)  # Add a batch dimension

        # Apply your emotion recognition model to the preprocessed face image
        predicted_emotion = model.predict(face_img)

        # Get the recognized emotion label
        emotion_label = emotion_labels[np.argmax(predicted_emotion)]
        # Get the confidence score (percentage) for the recognized emotion
        confidence = np.max(predicted_emotion) * 100

        # Display the emotion label and confidence near the face
        text = f"Emotion: {emotion_label} ({confidence:.2f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with emotion recognition in real time
    cv2.imshow('Emotion Recognition', frame)

    # Press 'q' to exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
