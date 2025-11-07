# OpenCV Starter Pack

# Table of Contents
1. [Overview](#overview)
2. [What is OpenCV](#what-is-opencv)
3. [Goal](#goal)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Demo](#1-demo-module-demopy)
7. [Model](#2-model-module-modelpy)
8. [Training](#3-training-module-trainpy)
9. [Preprocessing](#4-preprocessing-module-preprocesspy)
10. [Resources](#-resources)

## Overview
Hey hackers!

Welcome to the OpenCV starter pack. This repository and the quick guide associated with it is not meant to be a comprehensive guide on OpenCV or machine learning by any means, but it should provide some necessary modules to help you get started. This guide will cover **facial detection, basics of a convolutional neural network, training process, and real-time emotion detection.**

## What is OpenCV?
Have you ever wondered how the face ID on your phone works? Or how self-driving cars are able to process large amounts of noisy stimuli and still manage to (mostly) follow the laws of the road? All of this is made possible through an open-source library called OpenCV. It contains over 2500 algorithms related to real-time image processing and it serves as the backbone for a majority of computer vision applications. 

## Goal
This starter pack should teach you how to use OpenCV along with a machine learning model to create a real-time emotion detector. Hopefully by the end of this guide you will have a general idea of how to use OpenCV with your own projects and have a template to work from. Here are some possible hackathon ideas to give you some inspiration!
- **Mood-based music player**: Change music based on detected emotion
- **Emotion-aware chatbot**: Adjust responses based on user's mood
- **Accessibility tool**: Help people with autism recognize emotions
- **Gaming integration**: Control game mechanics with facial expressions
- **Mental health tracker**: Monitor emotional patterns over time

## Quick Start

We recommend the use of Anaconda to create and manage your virtual environment. For information on how to download Anaconda and set up a Conda environment, look [here](https://s4.ad.brown.edu/python2020/software.html). You can stop before the **Channels** section. 

### 1. Clone this repo
```bash
git clone https://github.com/hackatbrown/opencv-starter.git
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### *!! DISCLAIMER FOR MAC USERS !!*
If you have issues installing the dependencies on MacOS, try the following commands to set up a virtual environment and downgrade your Python installation to a supported version for Tensorflow:
```bash
brew install python@3.11
python3.11 -m venv ~/.venvs/opencv-starter
source ~/.venvs/opencv-starter/bin/activate
pip install -r ./requirements.txt
```

### 3. Run the Demo
```bash
python demo.py 
```

## 4. Train with your own data (optional)
If you would like to use the model in this database to detect your own custom emotions, upload your data as images in the same format as the file structure below under the data/ folder. This splits your data into a training set and testing set for the model – it is common to use an 80/20 split for the number of images that goes into each set, with more images going into training. The pretrained model included in this codebase was trained on the FER dataset, a public collection of facial expression images, organized by facial expression. If you want to use it yourself, you can find that [here](https://www.kaggle.com/datasets/msambare/fer2013).

## Project Structure

```
opencv-starter/
├── requirements.txt           # Python dependencies
├── demo.py                    # Main real-time demo with pre-trained model
├── train.py                   # Script to train an emotion detection model
├── preprocess.py              # Image preprocessing utilities
├── model.py                   # Model architecture definition
├── utils.py                   # General utility functions
├── data/                      # Data directory (training & test images)
│   ├── train/                 # Training images (FER & custom)
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── angry/
│   │   ├── excited/           # Example custom emotion
│   │   ├── confused/          # Example custom emotion
│   │   └── ...
│   └── test/                  # Testing images (FER test set)
│       ├── happy/
│       ├── sad/
│       └── ...
└── models/                    # Saved models and metadata
    ├── best_cnn_model.keras
    └── ...
```

# Module Guide

## 1. Demo Module (`demo.py`)

The demo module is where everything comes together - it's the real-time emotion detection application that uses both OpenCV and a pretrained model (or your model if you want to train your own).

**Video Capture:**
```python
cap = cv2.VideoCapture(0)
```
- `cv2.VideoCapture(0)` opens a connection to your webcam
- Think of this as opening a "stream" of images from your camera
- Each frame is a single image captured at that moment

**Face Detection with Haar Cascades:**
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
```

**What are Haar Cascades?**
- Haar Cascades are a machine learning-based approach for object detection
- They use "features" (patterns of light/dark regions) to identify faces
- Think of it like a smart filter that looks for patterns: two eyes, a nose, and a mouth in a certain arrangement
- The cascade part means it's a series of filters that get progressively more strict

**Image Processing:**
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
- `cv2.cvtColor()`: Converts from BGR (Blue-Green-Red, OpenCV's default) to grayscale
- Grayscale is used because Haar Cascades work on grayscale images (faster processing)

**Drawing on Images:**
```python
cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x, y - 10), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```
- `cv2.rectangle()`: Draws a rectangle around the detected face
- `cv2.putText()`: Overlays text (emotion label and confidence) on the frame
- These are OpenCV's drawing functions that modify the image data

**Display Loop:**
```python
cv2.imshow("Emotion Detection", frame)
cv2.waitKey(1)
```
- `cv2.imshow()`: Displays the processed frame in a window
- The loop runs continuously, processing ~30 frames per second

### General Flow:
1. **Capture** → Get frame from webcam
2. **Convert** → BGR to grayscale
3. **Detect** → Find faces using Haar Cascades
4. **Preprocess** → Prepare face region for model
5. **Predict** → Get emotion from CNN model
6. **Visualize** → Draw boxes and labels
7. **Display** → Show result and repeat

---

## 2. Model Module (`model.py`)

This module defines the Convolutional Neural Network (CNN) architecture that learns to recognize emotions from facial images. This is nowhere near a complete guide on deep learning, but for more information you might find this [tutorial](https://www.geeksforgeeks.org/deep-learning/deep-learning-tutorial/) useful.

### What is a CNN?

A CNN is a type of neural network designed for image recognition. It is composed of multiple layers that each look for different patterns:
- **Early layers**: Detect simple features (edges, corners)
- **Middle layers**: Detect complex patterns (eyes, mouth shapes)
- **Later layers**: Recognize full facial expressions (smile, frown)

### Architecture Breakdown:

**Block 1 - Basic Feature Detection:**
```python
layers.Conv2D(32, (3, 3), ...)  # 32 filters, each 3x3 pixels
```
- **Conv2D (Convolutional Layer)**: Slides a small "filter" (3x3 window) across the image
- Creates 32 different "feature maps" - each looking for different patterns
- Like having 32 different "lenses" that each detect something specific

**Batch Normalization:**
- Normalizes the outputs to have consistent ranges
- Helps training be more stable and faster
- Standardizing data so the network learns better

**MaxPooling:**
```python
layers.MaxPooling2D((2, 2))
```
- Takes the maximum value from each 2x2 region
- Reduces image size by half (48x48 → 24x24)
- Makes the network more efficient and helps it recognize patterns regardless of exact position

**Dropout:**
```python
layers.Dropout(0.25)
```
- Randomly "turns off" 25% of neurons during training
- Prevents overfitting (memorizing training data instead of learning patterns)
- Makes sure the model learns "conceptually" rather than memorizing the training data

**Block 2 & 3:**
- Same pattern but with more filters (64, then 128)
- Each block detects more complex patterns
- Image gets progressively smaller but more "abstract"

**Dense Layers (Final Classification):**
```python
layers.Flatten()  # Converts 2D feature maps to 1D
layers.Dense(512, activation='relu')  # Fully connected layer
layers.Dense(num_classes, activation='softmax')  # Final prediction
```
- **Flatten**: Converts the 2D feature maps into a 1D list
- **Dense layers**: Fully connected neurons that make the final decision
- **Softmax**: Converts raw scores into probabilities (all emotions sum to 100%)

### **Why This Architecture?**
- **Convolutional layers**: Detect spatial patterns (features that appear together)
- **Pooling**: Makes the network robust to small shifts/rotations
- **Dropout**: Prevents memorization
- **Multiple blocks**: Learns hierarchical features (simple → complex)

---

## 3. Training Module (`train.py`)

Now that we have our model, it's time to train! The training module orchestrates the entire machine learning pipeline, from loading data to saving a trained model.

### Training Pipeline Overview:

**1. Data Discovery:**
```python
self.emotions, self.emotion_to_idx, self.idx_to_emotion = discover_emotions(train_dir, test_dir)
```
- Automatically finds all emotion categories in your data folders
- Creates mappings: emotion name ↔ number (required for the model)

**2. Data Loading:**
```python
X_train, y_train, X_test, y_test = load_datasets(...)
```
- Loads images and labels from `data/train/` and `data/test/`
- Each image is preprocessed (resized, normalized) using OpenCV functions
- Returns numpy arrays ready for training

**3. Data Augmentation:**
```python
datagen = get_data_augmentation()
```
- Creates variations of training images: rotated, shifted, flipped
- Helps the model learn to recognize emotions in different conditions
- Provides a greater variety of training data

**4. Class Weights:**
```python
class_weights = compute_class_weight('balanced', ...)
```
- Handles imbalanced datasets (e.g., more "happy" than "disgust" images)

**5. Model Training:**
```python
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    class_weight=class_weight_dict
)
```

**Key Concepts:**
- **Batch**: Processes multiple images at once (64 at a time) for efficiency
- **Epoch**: One complete pass through all training data
- **Validation**: Tests on unseen data to check if model is learning correctly
- **Callbacks**: Automatically saves best model, stops early if no improvement, adjusts learning rate

**6. Model Saving:**
- Saves the trained model in `.keras` format
- Saves emotion mappings so you know which number = which emotion
- Saves metadata (accuracy, emotions, etc.)

---

### 4. Preprocessing Module (`preprocess.py`):
Image preprocessing is used for two reasons:
1. To ensure that the model receives data in the same format each time, no matter which image is used. 
2. To make processes that use image data such as training and live detection more efficient. 

**Image Preprocessing:**
```python
def preprocess_image(img_path, img_size=(48, 48), ...):
    img = cv2.imread(img_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, img_size)  # Resize to 48x48
    img = img.astype('float32') / 255.0  # Normalize to 0-1
```

**Why Each Step?**

1. **Reading Images:**
   - `cv2.imread()`: Loads image from file into a numpy array
   - Images are stored as arrays of pixel values

2. **Grayscale Conversion:**
   - Color doesn't help emotion detection (facial expressions are about shape, not color)
   - Grayscale reduces data size (3 channels → 1 channel)

3. **Resizing:**
   - All images must be the same size for the CNN (48x48 pixels)
   - `cv2.resize()` uses interpolation to resize without losing too much detail
   - Smaller images = faster training

4. **Normalization:**
   - Converts pixel values from 0-255 to 0.0-1.0
   - Neural networks train better with normalized data

## Training Tips
1. **Collect diverse data**: Different lighting, angles, expressions
2. **Balance your dataset**: Equal number of images per emotion
3. **Quality over quantity**: 50 good images > 200 poor images
4. **Test regularly**: Validate your model during training

## 📚 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide/keras)

## Used for this guide
- https://fuyofulo.medium.com/real-time-facial-emotion-recognition-using-deep-learning-and-opencv-30a331d39cf1
- https://learnopencv.com/facial-emotion-recognition/
- https://www.kaggle.com/code/syedmaazml/facial-emotion-recognition-using-cnn/notebook#Building-the-CNN-Model

## License
MIT License - feel free to use this in your projects!
