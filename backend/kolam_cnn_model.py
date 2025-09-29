"""
Kolam Pattern Recognition CNN Model

This module implements a Convolutional Neural Network for recognizing and classifying
kolam (rangoli) patterns from the collected dataset.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KolamCNNModel:
    """
    CNN model for kolam pattern recognition and classification.
    """

    def __init__(self, img_height: int = 224, img_width: int = 224, num_classes: int = None):
        """
        Initialize the CNN model.

        Args:
            img_height: Height of input images
            img_width: Width of input images
            num_classes: Number of pattern classes (auto-detected if None)
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

        # Model configuration
        self.config = {
            'img_height': img_height,
            'img_width': img_width,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'validation_split': 0.2
        }

    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and preprocess the kolam dataset.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Tuple of (images, labels, class_names)
        """
        images = []
        labels = []
        class_names = []

        logger.info(f"Loading dataset from {dataset_path}")

        # Walk through the dataset directory structure
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # Get the relative path for class label
                        rel_path = os.path.relpath(root, dataset_path)
                        class_name = rel_path.replace(
                            '\\', '/').replace('/', '_')

                        # Load image
                        img_path = os.path.join(root, file)
                        img = tf.keras.preprocessing.image.load_img(
                            img_path,
                            target_size=(self.img_height, self.img_width)
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(
                            img)
                        images.append(img_array)
                        labels.append(class_name)
                        class_names.append(class_name)

                    except Exception as e:
                        logger.warning(f"Error loading image {file}: {str(e)}")
                        continue

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        class_names = list(set(class_names))

        logger.info(
            f"Loaded {len(images)} images from {len(class_names)} classes")
        return images, labels, class_names

    def preprocess_data(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data for training.

        Args:
            images: Raw image data
            labels: Corresponding labels

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Normalize pixel values to [0, 1]
        images = images.astype('float32') / 255.0

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)

        # Update num_classes if not set
        if self.num_classes is None:
            self.num_classes = len(np.unique(y_encoded))

        # Convert labels to categorical
        y_categorical = tf.keras.utils.to_categorical(
            y_encoded, self.num_classes)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            images, y_categorical,
            test_size=self.config['validation_split'],
            random_state=42,
            stratify=y_encoded
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def build_model(self) -> keras.Model:
        """
        Build the CNN model architecture.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Data Augmentation
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),

            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
                self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()]
        )

        self.model = model
        logger.info("CNN model built successfully")
        return model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
        """
        Train the CNN model.

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels

        Returns:
            Training history
        """
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'backend/models/kolam_cnn_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Model training completed")
        return self.history

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model.

        Args:
            X_test: Test images
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0)

        # Generate classification report
        # Use actual unique classes present in test set to avoid mismatch
        unique_classes = np.unique(np.concatenate(
            [y_true_classes, y_pred_classes]))
        class_names = [str(i) for i in unique_classes]
        report = classification_report(
            y_true_classes, y_pred_classes,
            labels=unique_classes,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        logger.info(
            f"Model evaluation completed. Accuracy: {test_accuracy:.4f}")
        return results

    def save_model(self, model_path: str = 'backend/models/kolam_cnn_model.h5'):
        """
        Save the trained model.

        Args:
            model_path: Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str = 'backend/models/kolam_cnn_model.h5'):
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

    def predict(self, image_path: str) -> Dict:
        """
        Predict the class of a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))

        # Get class name
        class_names = list(self.label_encoder.classes_)
        predicted_label = class_names[predicted_class] if predicted_class < len(
            class_names) else str(predicted_class)

        return {
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        }

    def plot_training_history(self):
        """
        Plot the training history.
        """
        if self.history is None:
            logger.warning("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'],
                        label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'],
                        label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()

        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'],
                        label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()

        # Precision plot
        axes[1, 0].plot(self.history.history['precision'],
                        label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'],
                        label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()

        # Recall plot
        axes[1, 1].plot(self.history.history['recall'],
                        label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'],
                        label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('backend/models/training_history.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to train the CNN model.
    """
    # Initialize model
    model = KolamCNNModel(img_height=224, img_width=224)

    # Load dataset
    dataset_path = 'backend/rangoli_dataset_complete/images'
    images, labels, class_names = model.load_dataset(dataset_path)

    if len(images) == 0:
        logger.error("No images found in dataset")
        return

    # Preprocess data
    X_train, X_test, y_train, y_test = model.preprocess_data(images, labels)

    # Build model
    cnn_model = model.build_model()
    print(cnn_model.summary())

    # Train model
    history = model.train_model(X_train, y_train, X_test, y_test)

    # Evaluate model
    results = model.evaluate_model(X_test, y_test)

    # Plot training history
    model.plot_training_history()

    # Save model
    model.save_model()

    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Class names: {list(model.label_encoder.classes_)}")


if __name__ == "__main__":
    main()
