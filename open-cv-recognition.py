import cv2
import numpy as np
import joblib

# Load your trained k-NN model and the preprocessing components
knn_classifier = joblib.load('knn_model.pkl')  # Load your saved k-NN model
scaler = joblib.load('scaler.pkl')  # Load your saved StandardScaler
label_encoder = joblib.load('label_encoder.pkl')  # Load your saved LabelEncoder

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Increase the size of the bounding box
        x -= 10  # Move the left edge to the left
        y -= 10  # Move the top edge upward
        w += 20  # Increase the width
        h += 20  # Increase the height

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region and preprocess it
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48))
        flattened_face = face_resized.flatten()
        normalized_face = scaler.transform([flattened_face])

        # Make a prediction using the k-NN classifier
        predicted_label = knn_classifier.predict(normalized_face)
        emotion_label = label_encoder.inverse_transform(predicted_label)[0]

        # Display the emotion label on the frame
        cv2.putText(frame, "Emotion: " + emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with face detection and emotion recognition
    cv2.imshow('Emotion Recognition', frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
