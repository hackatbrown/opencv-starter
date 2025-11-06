#!/usr/bin/env python3
"""
Universal Emotion Detection Model Training
Trains a CNN model on all emotions found in data/train and data/test directories
Supports both FER dataset and custom emotions
"""

import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from dataset import discover_emotions, load_datasets
from model import create_model
from training_utils import get_training_callbacks, get_data_augmentation, print_data_statistics, save_model


class EmotionTrainer:
    """Universal emotion detection trainer that works with any emotion dataset"""
    
    def __init__(self, train_dir='./data/train', test_dir='./data/test', img_size=(48, 48)):
        self.img_size = img_size
        self.train_dir = train_dir
        self.test_dir = test_dir
        
        # Discover emotions from directories
        self.emotions, self.emotion_to_idx, self.idx_to_emotion = discover_emotions(train_dir, test_dir)
        
        if not self.emotions:
            print("No emotion data found!")
    
    
    def train(self, max_samples_per_emotion=2000, epochs=100, batch_size=64):
        """Main training function"""
        print("Starting emotion model training...")
        
        if not self.emotions:
            print("No emotions found!")
            return None
        
        # Load data
        X_train, y_train, X_test, y_test = load_datasets(
            self.train_dir, self.test_dir, 
            emotions=self.emotions,
            img_size=self.img_size,
            max_samples_per_emotion=max_samples_per_emotion
        )
        
        if X_train is None:
            return None
        
        # Convert string labels to integers
        y_train = np.array([self.emotion_to_idx[label] for label in y_train])
        y_test = np.array([self.emotion_to_idx[label] for label in y_test])
        
        # Print data statistics
        print_data_statistics(X_train, X_test, y_train, self.emotions)
        
        # Create model
        model = create_model(len(self.emotions), input_shape=(*self.img_size, 1))
        
        # Prepare data
        y_train_cat = to_categorical(y_train, num_classes=len(self.emotions))
        y_test_cat = to_categorical(y_test, num_classes=len(self.emotions))
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"Class weights: {class_weight_dict}")
        
        # Setup training
        datagen = get_data_augmentation()
        callbacks = get_training_callbacks()
        
        print("Starting training...")
        history = model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test_cat),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)[1]
        
        results = {
            'model': model,
            'accuracy': test_accuracy,
            'history': history,
            'emotion_to_idx': self.emotion_to_idx,
            'idx_to_emotion': self.idx_to_emotion
        }
        
        print(f"CNN model accuracy: {test_accuracy:.3f}")
        
        # Save model
        save_model(
            model, 
            self.emotions, 
            self.emotion_to_idx, 
            self.idx_to_emotion, 
            test_accuracy
        )
        
        return results


def main():
    print("Starting Emotion Model Training...")
    
    trainer = EmotionTrainer()
    
    if not trainer.emotions:
        print("No emotions found in data directories!")
        return
    
    results = trainer.train(max_samples_per_emotion=2000)
    
    if results:
        print("\nTraining completed successfully!")
        print("Model saved as: models/emotion_model.keras")
        print(f"Trained on {len(trainer.emotions)} emotions: {', '.join(trainer.emotions)}")
        print("You can now use this model for emotion detection!")
    else:
        print("Training failed. Please check your data and try again.")


if __name__ == "__main__":
    main()