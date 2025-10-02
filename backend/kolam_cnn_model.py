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
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
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
            'test_split': 0.2,  # Renamed from validation_split for clarity
            'use_multi_channel': True,
            'num_input_channels': 6  # RGB + Edges + Enhanced + Features
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
            if not files:
                continue
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # Get the relative path for class label
                        rel_path = os.path.relpath(root, dataset_path)
                        class_name = rel_path.replace(
                            os.sep, '_')  # Use os.sep for cross-platform compatibility

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

                    except Exception as e:
                        logger.warning(
                            f"Error loading image {img_path}: {str(e)}")
                        continue

        if not images:
            return np.array([]), np.array([]), []

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        class_names = list(sorted(np.unique(labels)))

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

        # Create multi-channel images if configured
        if self.config['use_multi_channel']:
            logger.info("Creating multi-channel images...")
            images = self._create_multi_channel_images(images)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)

        # Update num_classes if not set
        if self.num_classes is None:
            self.num_classes = len(np.unique(y_encoded))
            self.config['num_classes'] = self.num_classes

        # Convert labels to categorical
        y_categorical = tf.keras.utils.to_categorical(
            y_encoded, self.num_classes)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            images, y_categorical,
            test_size=self.config['test_split'],
            random_state=42,
            stratify=y_encoded
        )

        logger.info(
            f"Data split: Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # FIX 2: Removed the duplicated method definition. This is the single, correct version.
    def _create_edge_map(self, image: np.ndarray) -> np.ndarray:
        """
        Create edge map using Sobel operator.

        Args:
            image: Input RGB image (H, W, 3) in uint8 format

        Returns:
            Edge map (H, W, 1) normalized to [0, 1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        edge_map = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize to [0, 1]
        max_val = np.max(edge_map)
        if max_val > 0:
            edge_map = edge_map / max_val

        return edge_map.astype('float32')

    def _create_enhanced_luminance(self, image: np.ndarray) -> np.ndarray:
        """
        Create enhanced luminance using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input RGB image (H, W, 3)

        Returns:
            Enhanced luminance (H, W, 1) normalized to [0, 1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Normalize to [0, 1]
        enhanced = enhanced.astype('float32') / 255.0

        return enhanced

    def _create_feature_map(self, image: np.ndarray) -> np.ndarray:
        """
        Create handcrafted feature map using Laplacian of Gaussian.

        Args:
            image: Input RGB image (H, W, 3)

        Returns:
            Feature map (H, W, 1) normalized to [0, 1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        # Compute gradient magnitude using Laplacian
        feature_map = np.abs(laplacian)

        # Normalize to [0, 1]
        max_val = np.max(feature_map)
        if max_val > 0:
            feature_map = feature_map / max_val

        return feature_map.astype('float32')

    def _create_multi_channel_images(self, images: np.ndarray) -> np.ndarray:
        """
        Create 6-channel images by combining RGB + Edges + Enhanced + Features.

        Args:
            images: Input RGB images (N, H, W, 3) normalized to [0, 1]

        Returns:
            Multi-channel images (N, H, W, 6)
        """
        logger.info(
            f"Creating multi-channel images from {len(images)} input images...")

        multi_channel_images = []
        for i, image in enumerate(images):
            # Convert to uint8 for OpenCV operations
            image_uint8 = (image * 255).astype('uint8')

            # Create additional channels
            edge_map = self._create_edge_map(image_uint8)
            enhanced_luminance = self._create_enhanced_luminance(image_uint8)
            feature_map = self._create_feature_map(image_uint8)

            # Stack channels: RGB + Edges + Enhanced + Features
            multi_channel = np.concatenate([
                image,
                edge_map[..., np.newaxis],
                enhanced_luminance[..., np.newaxis],
                feature_map[..., np.newaxis]
            ], axis=-1)

            multi_channel_images.append(multi_channel)

            if (i + 1) % 100 == 0 or (i + 1) == len(images):
                logger.info(f"Processed {i + 1}/{len(images)} images")

        result = np.array(multi_channel_images)
        logger.info(
            f"Multi-channel preprocessing completed. Output shape: {result.shape}")
        return result

    def build_model(self) -> keras.Model:
        """
        Build the CNN model architecture.

        Returns:
            Compiled Keras model
        """
        input_channels = self.config['num_input_channels'] if self.config['use_multi_channel'] else 3

        model = keras.Sequential([
            layers.InputLayer(input_shape=(
                self.img_height, self.img_width, input_channels)),
            # Data Augmentation
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),

            # Convolutional Blocks
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
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

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()]
        )

        self.model = model
        logger.info(
            f"CNN model built successfully with {input_channels} input channels")
        return model

    # FIX 1 & 4: Corrected the train_model method to remove data leakage.
    # It now uses `validation_split` on the training data, as is best practice.
    # The separate validation set parameters have been removed.
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> keras.callbacks.History:
        """
        Train the CNN model.

        Args:
            X_train: Training images
            y_train: Training labels

        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'backend/models/kolam_cnn_best.h5', monitor='val_accuracy',
                save_best_only=True, mode='max'
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=0.2,  # Use a portion of training data for validation
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
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0)

        # FIX 3: Use the label encoder to get human-readable class names for the report.
        unique_class_indices = np.unique(
            np.concatenate([y_true_classes, y_pred_classes]))
        class_names = self.label_encoder.inverse_transform(
            unique_class_indices)

        report = classification_report(
            y_true_classes, y_pred_classes,
            labels=unique_class_indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(y_true_classes, y_pred_classes,
                              labels=np.arange(self.num_classes))

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
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str = 'backend/models/kolam_cnn_model.h5'):
        """
        Load a trained model.
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

    def predict(self, image_path: str) -> Dict:
        """
        Predict the class of a single image.
        """
        if self.model is None or self.label_encoder.classes_ is None:
            raise ValueError(
                "Model or LabelEncoder not initialized. Train or load a model first.")

        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        if self.config['use_multi_channel']:
            img_array = self._create_multi_channel_images(
                np.expand_dims(img_array, axis=0))[0]

        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))

        predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[
            0]

        return {
            'predicted_class_index': int(predicted_class_idx),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        }

    def plot_training_history(self):
        """
        Plot the training history and save it to a file.
        """
        if self.history is None:
            logger.warning("No training history available to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training and Validation Metrics', fontsize=16)

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'],
                        label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'],
                        label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'],
                        label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        # Check if precision metric exists
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history.get(
                'precision', []), label='Training Precision')
            axes[1, 0].plot(self.history.history.get(
                'val_precision', []), label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Recall
        # Check if recall metric exists
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history.get(
                'recall', []), label='Training Recall')
            axes[1, 1].plot(self.history.history.get(
                'val_recall', []), label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs('backend/models', exist_ok=True)
        plt.savefig('backend/models/training_history.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to train and evaluate the CNN model.
    """
    model = KolamCNNModel(img_height=224, img_width=224)

    dataset_path = 'backend/rangoli_dataset_complete/images'
    images, labels, class_names = model.load_dataset(dataset_path)

    if images.size == 0:
        logger.error(
            f"No images found in the specified dataset path: {dataset_path}")
        return

    X_train, X_test, y_train, y_test = model.preprocess_data(images, labels)

    cnn_model = model.build_model()
    cnn_model.summary()

    # FIX 1 & 4 (in action): Call the corrected train_model function.
    # It no longer requires the test set, preventing data leakage.
    model.train_model(X_train, y_train)

    logger.info("Evaluating model on the held-out test set...")
    results = model.evaluate_model(X_test, y_test)

    model.plot_training_history()

    model.save_model()

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Number of classes: {model.num_classes}")
    print("\nClassification Report:")
    # Pretty print the classification report dictionary
    print(json.dumps(results['classification_report'], indent=2))
    print("\n" + "="*50)


if __name__ == "__main__":
    main()
