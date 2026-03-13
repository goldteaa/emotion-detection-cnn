from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import cv2
import numpy as np
import os

# Load face detector and emotion recognition model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Function to detect emotion in an image
def detect_emotion(image_path):
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image not found or invalid path.")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        cv2.imshow('Emotion Detector', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the result
    cv2.imshow('Emotion Detector', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r'C:\Users\user\Desktop\Emotion_Detection_CNN-main\test.jpg'
detect_emotion(image_path)
