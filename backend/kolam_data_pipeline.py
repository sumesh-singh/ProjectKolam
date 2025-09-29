"""
Kolam Data Preprocessing Pipeline

This module provides comprehensive data preprocessing utilities for both CNN and GAN models,
including data loading, augmentation, normalization, and dataset splitting.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KolamDataPipeline:
    """
    Comprehensive data preprocessing pipeline for kolam pattern recognition and generation.
    """

    def __init__(self, img_height: int = 224, img_width: int = 224):
        """
        Initialize the data pipeline.

        Args:
            img_height: Target image height
            img_width: Target image width
        """
        self.img_height = img_height
        self.img_width = img_width
        self.label_encoder = LabelEncoder()

        # Data storage
        self.images = None
        self.labels = None
        self.metadata = None
        self.class_names = None
        self.num_classes = None

        # Preprocessing configuration
        self.config = {
            'validation_split': 0.2,
            'test_split': 0.1,
            'random_seed': 42,
            'apply_augmentation': True,
            'normalize_images': True,
            'balance_classes': False
        }

    def load_dataset(self, dataset_path: str, metadata_path: str = None) -> Dict:
        """
        Load the complete dataset with images and metadata.

        Args:
            dataset_path: Path to the images directory
            metadata_path: Path to metadata directory (optional)

        Returns:
            Dictionary containing dataset information
        """
        logger.info(f"Loading dataset from {dataset_path}")

        images = []
        labels = []
        metadata_list = []
        class_folders = {}

        # Walk through the dataset directory
        for root, dirs, files in os.walk(dataset_path):
            class_name = os.path.relpath(root, dataset_path).replace(
                '\\', '/').replace('/', '_')

            if class_name == '.':
                continue

            image_files = [f for f in files if f.lower().endswith(
                ('.jpg', '.jpeg', '.png'))]

            if image_files:
                class_folders[class_name] = len(image_files)

                for file in image_files:
                    try:
                        img_path = os.path.join(root, file)

                        # Load image
                        img = tf.keras.preprocessing.image.load_img(
                            img_path,
                            target_size=(self.img_height, self.img_width)
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(
                            img)
                        images.append(img_array)

                        # Determine label based on folder structure
                        labels.append(class_name)

                        # Load metadata if available
                        if metadata_path:
                            metadata_file = os.path.join(
                                metadata_path, os.path.splitext(file)[0] + '.json')
                            if os.path.exists(metadata_file):
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    metadata_list.append(metadata)

                    except Exception as e:
                        logger.warning(f"Error loading image {file}: {str(e)}")
                        continue

        # Convert to numpy arrays
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.metadata = metadata_list if metadata_list else None

        # Get class information
        self.class_names = list(class_folders.keys())
        self.num_classes = len(self.class_names)

        # Encode labels
        if len(self.labels) > 0:
            self.labels = self.label_encoder.fit_transform(self.labels)

        dataset_info = {
            'total_images': len(images),
            'num_classes': self.num_classes,
            'class_distribution': class_folders,
            'image_shape': (self.img_height, self.img_width, 3),
            'class_names': self.class_names
        }

        logger.info(
            f"Dataset loaded: {dataset_info['total_images']} images, {dataset_info['num_classes']} classes")
        return dataset_info

    def create_data_augmentation(self) -> tf.keras.Sequential:
        """
        Create data augmentation pipeline.

        Returns:
            Keras Sequential model for data augmentation
        """
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomTranslation(0.1, 0.1),
        ])

        return data_augmentation

    def normalize_images(self, images: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize images using different methods.

        Args:
            images: Input images
            method: Normalization method ('standard', 'minmax', 'tanh')

        Returns:
            Normalized images
        """
        if method == 'standard':
            # Standard normalization (0-1 range)
            normalized = images.astype('float32') / 255.0
        elif method == 'minmax':
            # Min-max normalization
            normalized = (images - np.min(images)) / \
                (np.max(images) - np.min(images))
        elif method == 'tanh':
            # Tanh normalization (-1 to 1 range)
            normalized = (images.astype('float32') - 127.5) / 127.5
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def balance_dataset(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset using oversampling or undersampling.

        Args:
            images: Input images
            labels: Input labels

        Returns:
            Tuple of balanced (images, labels)
        """
        logger.info("Balancing dataset...")

        # Count class distribution
        class_counts = Counter(labels)
        max_count = max(class_counts.values())

        balanced_images = []
        balanced_labels = []

        for class_label in np.unique(labels):
            # Get indices for this class
            class_indices = np.where(labels == class_label)[0]

            # Add original images
            class_images = images[class_indices]
            balanced_images.extend(class_images)
            balanced_labels.extend([class_label] * len(class_images))

            # Calculate how many more images needed
            needed = max_count - len(class_images)

            if needed > 0:
                # Oversample by duplicating with augmentation
                for _ in range(needed // len(class_images)):
                    balanced_images.extend(class_images)
                    balanced_labels.extend([class_label] * len(class_images))

                # Add remaining samples
                remaining = needed % len(class_images)
                if remaining > 0:
                    balanced_images.extend(class_images[:remaining])
                    balanced_labels.extend([class_label] * remaining)

        return np.array(balanced_images), np.array(balanced_labels)

    def split_dataset(self, test_size: float = None, validation_size: float = None,
                      stratify: bool = True) -> Dict[str, np.ndarray]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            test_size: Size of test set (default: config value)
            validation_size: Size of validation set (default: config value)
            stratify: Whether to stratify by class labels

        Returns:
            Dictionary containing split datasets
        """
        if test_size is None:
            test_size = self.config['test_split']
        if validation_size is None:
            validation_size = self.config['validation_split']

        if self.images is None or self.labels is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        # First split: separate test set
        stratify_param = self.labels if stratify else None

        # Handle classes with too few samples
        if stratify:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            # Classes with at least 2 samples
            valid_classes = unique_labels[counts >= 2]

            if len(valid_classes) < len(unique_labels):
                # Use non-stratified split for classes with too few samples
                logger.warning(
                    "Some classes have too few samples for stratified split. Using random split.")
                stratify_param = None

        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels,
            test_size=test_size,
            random_state=self.config['random_seed'],
            stratify=stratify_param
        )

        # Second split: separate validation set from remaining data
        stratify_param = y_temp if stratify else None
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config['random_seed'],
            stratify=stratify_param
        )

        # Normalize images
        if self.config['normalize_images']:
            X_train = self.normalize_images(X_train, 'standard')
            X_val = self.normalize_images(X_val, 'standard')
            X_test = self.normalize_images(X_test, 'standard')

        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        logger.info(
            f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return splits

    def create_tf_datasets(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                           shuffle: bool = True, augment: bool = None) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with optional augmentation.

        Args:
            X: Input images
            y: Input labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation (default: config value)

        Returns:
            TensorFlow Dataset object
        """
        if augment is None:
            augment = self.config['apply_augmentation']

        # Convert labels to categorical
        y_categorical = tf.keras.utils.to_categorical(y, self.num_classes)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y_categorical))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))

        # Batch the data
        dataset = dataset.batch(batch_size)

        # Apply augmentation if requested
        if augment:
            data_augmentation = self.create_data_augmentation()

            def augment_fn(X, y):
                X = data_augmentation(X)
                return X, y

            dataset = dataset.map(
                augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def analyze_dataset(self) -> Dict:
        """
        Analyze the dataset and provide statistics.

        Returns:
            Dictionary containing dataset analysis
        """
        if self.images is None or self.labels is None:
            raise ValueError("No dataset loaded")

        if len(self.images) == 0:
            raise ValueError("Dataset is empty")

        # Basic statistics
        total_images = len(self.images)
        image_shape = self.images.shape[1:]

        # Class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))

        # Image statistics
        pixel_stats = {
            'mean': np.mean(self.images),
            'std': np.std(self.images),
            'min': np.min(self.images),
            'max': np.max(self.images)
        }

        # Class balance analysis
        min_class_count = min(counts)
        max_class_count = max(counts)
        imbalance_ratio = max_class_count / min_class_count

        analysis = {
            'total_images': total_images,
            'image_shape': image_shape,
            'num_classes': self.num_classes,
            'class_distribution': class_distribution,
            'pixel_statistics': pixel_stats,
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': imbalance_ratio < 2.0
        }

        return analysis

    def visualize_dataset(self, save_path: str = 'backend/dataset_analysis'):
        """
        Create visualizations of the dataset.

        Args:
            save_path: Path to save visualizations
        """
        os.makedirs(save_path, exist_ok=True)

        if self.images is None or self.labels is None:
            raise ValueError("No dataset loaded")

        # Class distribution plot
        plt.figure(figsize=(12, 6))
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        class_names_display = [self.class_names[label]
                               for label in unique_labels]

        plt.bar(range(len(unique_labels)), counts,
                tick_label=class_names_display)
        plt.xticks(rotation=45, ha='right')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.tight_layout()
        plt.savefig(f'{save_path}/class_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Sample images grid
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Sample Images from Dataset', fontsize=16)

        for i in range(16):
            if i < len(self.images):
                row = i // 4
                col = i % 4

                # Denormalize for display
                img = self.images[i]
                if np.max(img) <= 1.0:
                    img = img * 255.0
                img = img.astype('uint8')

                axes[row, col].imshow(img)
                axes[row, col].set_title(
                    f'Class: {self.class_names[self.labels[i]]}')
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_path}/sample_images.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Dataset visualizations saved to {save_path}")

    def prepare_for_cnn(self, batch_size: int = 32) -> Dict:
        """
        Prepare dataset specifically for CNN training.

        Args:
            batch_size: Batch size for datasets

        Returns:
            Dictionary containing prepared datasets and information
        """
        # Split dataset
        splits = self.split_dataset()

        # Create TensorFlow datasets
        train_dataset = self.create_tf_datasets(
            splits['X_train'], splits['y_train'],
            batch_size=batch_size, shuffle=True
        )

        val_dataset = self.create_tf_datasets(
            splits['X_val'], splits['y_val'],
            batch_size=batch_size, shuffle=False
        )

        test_dataset = self.create_tf_datasets(
            splits['X_test'], splits['y_test'],
            batch_size=batch_size, shuffle=False
        )

        # Calculate class weights for imbalanced datasets
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(splits['y_train']),
            y=splits['y_train']
        )
        class_weights = dict(enumerate(class_weights))

        preparation_info = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'class_weights': class_weights,
            'num_train_batches': len(splits['X_train']) // batch_size,
            'num_val_batches': len(splits['X_val']) // batch_size,
            'num_test_batches': len(splits['X_test']) // batch_size,
            'splits': splits
        }

        return preparation_info

    def prepare_for_gan(self) -> np.ndarray:
        """
        Prepare dataset specifically for GAN training.

        Returns:
            Preprocessed images for GAN training
        """
        if self.images is None:
            raise ValueError("No dataset loaded")

        # Normalize for GAN (tanh activation)
        processed_images = self.normalize_images(self.images, method='tanh')

        return processed_images


def main():
    """
    Main function to demonstrate the data pipeline.
    """
    # Initialize pipeline
    pipeline = KolamDataPipeline(img_height=128, img_width=128)

    # Load dataset
    dataset_path = 'backend/rangoli_dataset_complete/images'
    metadata_path = 'backend/rangoli_dataset_complete/metadata'

    dataset_info = pipeline.load_dataset(dataset_path, metadata_path)
    print("Dataset Info:", dataset_info)

    # Analyze dataset
    analysis = pipeline.analyze_dataset()
    print("Dataset Analysis:", analysis)

    # Create visualizations
    pipeline.visualize_dataset()

    # Prepare for CNN
    cnn_data = pipeline.prepare_for_cnn(batch_size=32)
    print(
        f"CNN datasets prepared: {cnn_data['num_train_batches']} train batches")

    # Prepare for GAN
    gan_data = pipeline.prepare_for_gan()
    print(f"GAN data prepared: {gan_data.shape}")


if __name__ == "__main__":
    main()
