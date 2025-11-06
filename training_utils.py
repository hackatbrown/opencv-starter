#!/usr/bin/env python3
"""
Training utility functions for emotion detection model training.
Handles callbacks, data augmentation, and training configuration.
"""

import os
import numpy as np
import pickle
import json
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_training_callbacks(checkpoint_path='models/best_cnn_model.keras', 
                          early_stop_patience=10, 
                          lr_patience=5,
                          monitor='val_loss'):
    """
    Get training callbacks for model training.
    """
    return [
        EarlyStopping(monitor=monitor, patience=early_stop_patience, restore_best_weights=True), 
        ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=lr_patience, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
    ]


def get_data_augmentation(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest'):
    """
    Get data augmentation generator configuration.
    """
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )


def print_data_statistics(X_train, X_test, y_train, emotions):
    """
    Print data statistics including class distribution.
    
    Args:
        X_train: Training images
        X_test: Test images
        y_train: Training labels (integer encoded)
        emotions: List of emotion names
    """
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    print(f"Data shape: {X_train.shape}")
    print(f"Data range: {X_train.min():.3f} to {X_train.max():.3f}")
    
    # Class distribution
    class_counts = np.bincount(y_train)
    for i, emotion in enumerate(emotions):
        if i < len(class_counts):
            print(f"  {emotion}: {class_counts[i]} samples")
    
    # Check for class imbalance
    if len(class_counts) > 0:
        min_count = np.min(class_counts)
        max_count = np.max(class_counts)
        if max_count > min_count * 10:
            print(f"Severe class imbalance! Min: {min_count}, Max: {max_count}")


def save_model(model, emotions, emotion_to_idx, idx_to_emotion, accuracy=0, 
               model_path='models/universal_emotion_model.keras'):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained Keras model
        emotions: List of emotion names
        emotion_to_idx: Dictionary mapping emotion names to indices
        idx_to_emotion: Dictionary mapping indices to emotion names
        accuracy: Model accuracy
        model_path: Path to save the model
    """
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    if model:
        model.save(model_path)
        
        # Save emotion mappings
        mapping_path = os.path.join('models', 'universal_emotion_mappings.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'emotion_to_idx': emotion_to_idx,
                'idx_to_emotion': idx_to_emotion
            }, f)
        
        print("Model saved successfully!")
    
    # Save model info
    model_info = {
        'model_type': 'universal_cnn',
        'emotions': emotions,
        'total_classes': len(emotions),
        'dataset': 'mixed' if len(emotions) > 7 else 'fer_only',
        'accuracy': accuracy
    }
    
    info_path = os.path.join('models', 'universal_model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
