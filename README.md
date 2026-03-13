# Emotion Detection using Convolutional Neural Networks

## Project Overview

This project implements a **deep learning model for facial emotion recognition** using Convolutional Neural Networks (CNN).

The system detects faces from a webcam feed using **OpenCV** and predicts the emotion displayed by each detected face using a trained **Keras CNN model**.

Facial emotion recognition is an important application of **computer vision and artificial intelligence**, with use cases in human-computer interaction, behavioral analysis, and smart interfaces.

---

## Objective

The goal of this project is to build a real-time system capable of detecting human emotions from facial expressions using deep learning techniques.

The model analyzes facial images and classifies them into multiple emotional categories, demonstrating how convolutional neural networks can be applied to real-world computer vision problems.

---

## Dataset

The model was trained using the **FER2013 Facial Expression Dataset**, a widely used dataset for emotion recognition tasks.

The dataset contains thousands of grayscale facial images labeled with different emotional expressions.

Dataset source:

https://www.kaggle.com/datasets/msambare/fer2013

---

## Tools & Technologies

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Convolutional Neural Networks (CNN)

---

## Methodology

1. Face Detection using OpenCV Haar Cascade
2. Image preprocessing and resizing
3. Training a Convolutional Neural Network using Keras
4. Emotion classification using the trained CNN model
5. Real-time emotion prediction using webcam input

The model processes facial images and predicts one of several emotional states.

---

## Results

The trained CNN model successfully detects and classifies emotions from facial expressions in real time.

The system combines **face detection with deep learning classification**, allowing the model to identify emotional states directly from webcam video streams.

---

## Project Structure

```
emotion-detection-cnn
│
├── main.py
├── predict_image.py
├── emotion-classification-cnn-using-keras.ipynb
├── haarcascade_frontalface_default.xml
├── model.h5
└── README.md
```

---

## How to Run

1. Clone the repository

```
git clone https://github.com/goldteaa/emotion-detection-cnn.git
```

2. Install dependencies

```
pip install opencv-python tensorflow keras numpy
```

3. Run real-time emotion detection

```
python main.py
```

4. Run emotion detection on an image

```
python predict_image.py
```

---

## Emotions Detected

The model classifies the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## Future Improvements

* Improve model accuracy using deeper CNN architectures
* Train the model on larger datasets
* Deploy the system as a web application
* Add support for emotion detection from video files

---
