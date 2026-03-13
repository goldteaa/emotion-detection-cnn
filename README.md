# Emotion Detection CNN

Real-time facial emotion recognition using **Convolutional Neural Networks (CNN)**, **OpenCV**, and **Keras**.

This project detects human emotions from facial expressions using a trained deep learning model and a webcam feed.

---

## Project Overview

Facial emotion recognition is an important application of computer vision and deep learning.  
This project implements a CNN-based model trained to classify facial expressions into different emotional categories.

The system uses **OpenCV** to detect faces from webcam input and a **Keras deep learning model** to predict the emotion of each detected face.

---

## Features

- Real-time emotion detection using webcam
- Face detection using Haar Cascade classifier
- CNN-based emotion classification
- Emotion prediction from images
- Deep learning model trained with Keras

---

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Convolutional Neural Networks (CNN)

---

## Project Structure
emotion-detection-cnn
│
├ main.py # Real-time webcam emotion detection
├ predict_image.py # Emotion detection from an image
├ emotion-classification-cnn-using-keras.ipynb # Model training notebook
├ haarcascade_frontalface_default.xml # Face detection model
├ model.h5 # Trained CNN model
└ README.md


---

## Dataset

The model was trained using the **FER2013 Facial Expression Dataset**.

Dataset link:  
https://www.kaggle.com/datasets/msambare/fer2013

---

## Installation

Clone the repository:
git clone https://github.com/goldteaa/emotion-detection-cnn.git

Install dependencies:
pip install opencv-python tensorflow keras numpy

---

## How to Run

Run real-time emotion detection:


python main.py


Run emotion detection on an image:


python predict_image.py


---

## Emotions Detected

The model classifies the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## Future Improvements

- Improve model accuracy with deeper CNN architectures
- Deploy as a web application
- Add support for video file emotion analysis
- Build a real-time emotion dashboard

---

## Author
AM
Created as part of a **Master's Degree in Data Science & Business Aalytics** project.
