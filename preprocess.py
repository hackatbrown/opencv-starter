#!/usr/bin/env python3
"""
Preprocessing functions for emotion detection models.
Handles image preprocessing, normalization, and feature extraction.
"""

import cv2
import numpy as np


def preprocess_face(gray_frame, box, target_size=(48, 48)):
    """
    Crop, resize, and normalize a face region for model prediction.
    """
    x, y, w, h = box
    face = gray_frame[y:y+h, x:x+w]
    if face.size == 0:
        return None
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)  # (48, 48, 1)
    face = np.expand_dims(face, axis=0)   # (1, 48, 48, 1)
    return face


def preprocess_image(img_path, img_size=(48, 48), return_batch=False):
    """
    Preprocess a single image file for model input.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        img = cv2.resize(img, img_size)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add channel dimension for CNN
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        
        if return_batch:
            img = np.expand_dims(img, axis=0)  # (1, H, W, 1)
        
        return img
    except Exception as e:
        return None


def preprocess_image_array(img_array, img_size=(48, 48), return_batch=False):
    """
    Preprocess an image array (numpy array) for model input.
    
    Args:
        img_array: Input image as numpy array (BGR or grayscale)
        img_size: Target size for resizing (width, height)
        return_batch: If True, returns batch format (1, H, W, 1), else (H, W, 1)
    
    Returns:
        Preprocessed image array
        Shape: (H, W, 1) or (1, H, W, 1) if return_batch=True
    """
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img = img_array.copy()
    
    # Resize to standard size
    img = cv2.resize(img, img_size)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add channel dimension for CNN
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    
    if return_batch:
        img = np.expand_dims(img, axis=0)  # (1, H, W, 1)
    
    return img
