#!/usr/bin/env python3
"""
Dataset loading utilities for emotion detection training.
Handles scanning directories, loading images, and preparing data.
"""

import os
import numpy as np
from preprocess import preprocess_image


def scan_emotion_directory(directory):
    """
    Scan a directory for emotion subdirectories with images.
    
    Args:
        directory: Directory path to scan
        
    Returns:
        Set of emotion names found
    """
    emotions = set()
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Check if it has images
                images = [f for f in os.listdir(item_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(images) > 0:
                    emotions.add(item)
    except OSError as e:
        print(f"Error scanning {directory}: {e}")
    
    return emotions


def discover_emotions(train_dir='./data/train', test_dir='./data/test'):
    """
    Discover all emotions from train and test directories.
    
    Args:
        train_dir: Training data directory
        test_dir: Test data directory
        
    Returns:
        Tuple of (sorted_emotions_list, emotion_to_idx_dict, idx_to_emotion_dict)
    """
    emotions = set()
    for directory in [train_dir, test_dir]:
        if directory and os.path.exists(directory):
            emotions.update(scan_emotion_directory(directory))
    
    if not emotions:
        return [], {}, {}
    
    emotions_list = sorted(list(emotions))
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions_list)}
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
    
    print(f"Found {len(emotions_list)} emotions: {', '.join(emotions_list)}")
    return emotions_list, emotion_to_idx, idx_to_emotion


def load_datasets(train_dir='./data/train', test_dir='./data/test', 
                  emotions=None, img_size=(48, 48), max_samples_per_emotion=2000):
    """
    Load training and test datasets.
    
    Args:
        train_dir: Training data directory
        test_dir: Test data directory
        emotions: List of emotions (if None, will auto-discover)
        img_size: Target image size
        max_samples_per_emotion: Maximum samples per emotion
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) or (None, None, None, None) if error
    """
    # Auto-discover emotions if not provided
    if emotions is None:
        emotions, _, _ = discover_emotions(train_dir, test_dir)
        if not emotions:
            print("No emotion data found!")
            return None, None, None, None
    
    def _load_from_directory(data_dir):
        """Helper to load images from a single directory"""
        images, labels = [], []
        for emotion in emotions:
            emotion_path = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_path):
                count = 0
                for img_file in os.listdir(emotion_path):
                    if count >= max_samples_per_emotion:
                        break
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_path, img_file)
                        img = preprocess_image(img_path, img_size=img_size, return_batch=False)
                        if img is not None:
                            images.append(img)
                            labels.append(emotion)
                            count += 1
                print(f"  {emotion}: {count} images")
        return (np.array(images) if images else None, 
                np.array(labels) if labels else None)
    
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    # Load training data
    if train_dir and os.path.exists(train_dir):
        dir_images, dir_labels = _load_from_directory(train_dir)
        if dir_images is not None:
            train_images.extend(dir_images)
            train_labels.extend(dir_labels)
            print(f"{train_dir}: {len(dir_images)} images")

    # Load test data
    if test_dir and os.path.exists(test_dir):
        dir_images, dir_labels = _load_from_directory(test_dir)
        if dir_images is not None:
            test_images.extend(dir_images)
            test_labels.extend(dir_labels)
            print(f"{test_dir}: {len(dir_images)} images")
    
    if len(train_images) == 0 or len(test_images) == 0:
        print("No training or test data found!")
        return None, None, None, None
    
    print(f"Total dataset: {len(train_images)} training images, {len(test_images)} test images")
    return (np.array(train_images), np.array(train_labels), 
            np.array(test_images), np.array(test_labels))
