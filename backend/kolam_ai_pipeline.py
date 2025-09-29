"""
Kolam AI Recognition and Recreation Pipeline

This module integrates the CNN and GAN models to provide a complete pipeline for
kolam pattern recognition and recreation, forming the core AI system.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Import our custom modules
from kolam_cnn_model import KolamCNNModel
from kolam_gan_model import KolamGAN
from kolam_data_pipeline import KolamDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KolamAIPipeline:
    """
    Integrated AI pipeline for kolam recognition and recreation.
    """

    def __init__(self, img_height: int = 224, img_width: int = 224, latent_dim: int = 100):
        """
        Initialize the AI pipeline.

        Args:
            img_height: Image height for CNN
            img_width: Image width for CNN
            latent_dim: Latent dimension for GAN
        """
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim

        # Initialize components
        self.data_pipeline = KolamDataPipeline(img_height, img_width)
        self.cnn_model = KolamCNNModel(img_height, img_width)
        self.gan_model = KolamGAN(img_height, img_width)

        # Pipeline state
        self.is_trained = False
        self.training_history = {}

        # Configuration
        self.config = {
            'cnn_epochs': 50,
            'gan_epochs': 1000,
            'batch_size': 32,
            'save_models': True,
            'model_dir': 'backend/models'
        }

    def prepare_dataset(self, dataset_path: str, metadata_path: str = None) -> Dict:
        """
        Prepare the dataset for both CNN and GAN training.

        Args:
            dataset_path: Path to the dataset
            metadata_path: Path to metadata (optional)

        Returns:
            Dictionary containing dataset preparation info
        """
        logger.info("Preparing dataset for AI pipeline...")

        # Load dataset using data pipeline
        dataset_info = self.data_pipeline.load_dataset(
            dataset_path, metadata_path)

        # Analyze dataset (only if images were loaded)
        if self.data_pipeline.images is not None and len(self.data_pipeline.images) > 0:
            analysis = self.data_pipeline.analyze_dataset()
        else:
            analysis = {'total_images': 0, 'num_classes': 0,
                        'error': 'No images loaded'}

        # Create visualizations
        self.data_pipeline.visualize_dataset()

        # Prepare data for CNN
        cnn_data = self.data_pipeline.prepare_for_cnn(
            batch_size=self.config['batch_size'])

        # Prepare data for GAN
        gan_data = self.data_pipeline.prepare_for_gan()

        preparation_info = {
            'dataset_info': dataset_info,
            'analysis': analysis,
            'cnn_data': cnn_data,
            'gan_data': gan_data,
            'total_samples': len(self.data_pipeline.images)
        }

        logger.info(
            f"Dataset prepared: {preparation_info['total_samples']} samples")
        return preparation_info

    def train_cnn_model(self, train_data: Dict, epochs: int = None) -> Dict:
        """
        Train the CNN model for pattern recognition.

        Args:
            train_data: Training data dictionary
            epochs: Number of epochs (optional)

        Returns:
            Training results dictionary
        """
        if epochs is None:
            epochs = self.config['cnn_epochs']

        logger.info("Training CNN model...")

        # Extract data
        X_train = train_data['splits']['X_train']
        y_train = train_data['splits']['y_train']
        X_val = train_data['splits']['X_val']
        y_val = train_data['splits']['y_val']

        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(
            y_train, self.data_pipeline.num_classes)
        y_val_cat = tf.keras.utils.to_categorical(
            y_val, self.data_pipeline.num_classes)

        # Update CNN model with correct number of classes
        self.cnn_model.num_classes = self.data_pipeline.num_classes

        # Build and train CNN model
        self.cnn_model.build_model()
        history = self.cnn_model.train_model(
            X_train, y_train_cat, X_val, y_val_cat)

        # Evaluate model
        results = self.cnn_model.evaluate_model(train_data['splits']['X_test'],
                                                tf.keras.utils.to_categorical(
            train_data['splits']['y_test'],
            self.data_pipeline.num_classes))

        # Save model if requested
        if self.config['save_models']:
            self.cnn_model.save_model(os.path.join(
                self.config['model_dir'], 'kolam_cnn_final.h5'))

        training_results = {
            'cnn_history': history.history,
            'cnn_evaluation': results,
            'epochs_trained': len(history.history['loss']),
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }

        logger.info(
            f"CNN training completed. Final accuracy: {training_results['final_val_accuracy']:.4f}")
        return training_results

    def train_gan_model(self, gan_data: np.ndarray, epochs: int = None) -> Dict:
        """
        Train the GAN model for pattern generation.

        Args:
            gan_data: GAN training data
            epochs: Number of epochs (optional)

        Returns:
            Training results dictionary
        """
        if epochs is None:
            epochs = self.config['gan_epochs']

        logger.info("Training GAN model...")

        # Create temporary dataset structure for GAN training
        temp_dataset_path = 'backend/temp_gan_dataset'
        os.makedirs(temp_dataset_path, exist_ok=True)

        # Save a subset of images for GAN training (to speed up training)
        # Use subset for faster training
        subset_size = min(1000, len(gan_data))
        subset_indices = np.random.choice(
            len(gan_data), subset_size, replace=False)
        gan_subset = gan_data[subset_indices]

        # Create temporary image files (GAN training expects file paths)
        for i, img in enumerate(gan_subset):
            # Convert from [-1, 1] back to [0, 255] for saving
            img_normalized = (img * 127.5 + 127.5).astype('uint8')
            img_pil = tf.keras.preprocessing.image.array_to_img(img_normalized)
            img_pil.save(f'{temp_dataset_path}/temp_{i}.jpg')

        # Train GAN
        self.gan_model.train(temp_dataset_path, epochs=epochs)

        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dataset_path)

        # Save models
        if self.config['save_models']:
            self.gan_model.save_models()

        training_results = {
            'gan_epochs_trained': epochs,
            'd_losses': self.gan_model.d_losses,
            'g_losses': self.gan_model.g_losses,
            'final_d_loss': self.gan_model.d_losses[-1] if self.gan_model.d_losses else None,
            'final_g_loss': self.gan_model.g_losses[-1] if self.gan_model.g_losses else None
        }

        logger.info("GAN training completed")
        return training_results

    def train_full_pipeline(self, dataset_path: str, metadata_path: str = None,
                            cnn_epochs: int = None, gan_epochs: int = None) -> Dict:
        """
        Train the complete AI pipeline.

        Args:
            dataset_path: Path to dataset
            metadata_path: Path to metadata
            cnn_epochs: CNN training epochs
            gan_epochs: GAN training epochs

        Returns:
            Complete training results
        """
        logger.info("Starting full AI pipeline training...")

        # Prepare dataset (fix path if needed)
        if not os.path.exists(dataset_path):
            # Try relative path
            if os.path.exists(dataset_path.replace('backend/', '')):
                dataset_path = dataset_path.replace('backend/', '')

        prep_info = self.prepare_dataset(dataset_path, metadata_path)

        # Train CNN model
        cnn_results = self.train_cnn_model(prep_info['cnn_data'], cnn_epochs)

        # Train GAN model
        gan_results = self.train_gan_model(prep_info['gan_data'], gan_epochs)

        # Update pipeline state
        self.is_trained = True
        self.training_history = {
            'dataset_info': prep_info,
            'cnn_results': cnn_results,
            'gan_results': gan_results,
            'training_timestamp': datetime.now().isoformat()
        }

        # Save training history
        self.save_training_history()

        logger.info("Full pipeline training completed successfully")
        return self.training_history

    def recognize_pattern(self, image_path: str) -> Dict:
        """
        Recognize a kolam pattern from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Recognition results dictionary
        """
        if not self.is_trained:
            raise ValueError(
                "Pipeline not trained. Call train_full_pipeline() first.")

        # Use CNN model for recognition
        results = self.cnn_model.predict(image_path)

        # Add additional information
        enhanced_results = {
            'recognition': results,
            'predicted_class_name': self.data_pipeline.class_names[results['predicted_class']] if results['predicted_class'] < len(self.data_pipeline.class_names) else 'Unknown',
            'confidence_percentage': results['confidence'] * 100,
            'timestamp': datetime.now().isoformat()
        }

        return enhanced_results

    def generate_kolam_patterns(self, num_patterns: int = 1, class_condition: int = None) -> List[Dict]:
        """
        Generate new kolam patterns.

        Args:
            num_patterns: Number of patterns to generate
            class_condition: Optional class conditioning (for future use)

        Returns:
            List of generated pattern dictionaries
        """
        if not self.is_trained:
            raise ValueError(
                "Pipeline not trained. Call train_full_pipeline() first.")

        logger.info(f"Generating {num_patterns} kolam patterns...")

        # Generate patterns using GAN
        generated_images = self.gan_model.generate_kolam(num_patterns)

        if generated_images is None:
            return []

        # Create response for each generated pattern
        patterns = []
        for i, img in enumerate(generated_images):
            pattern_info = {
                'pattern_id': f'generated_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{i}',
                'image_data': img.tolist(),  # Convert to list for JSON serialization
                'image_shape': img.shape,
                'generation_timestamp': datetime.now().isoformat(),
                'generation_method': 'gan',
                'class_condition': class_condition
            }
            patterns.append(pattern_info)

        return patterns

    def analyze_and_generate(self, input_image_path: str, num_variations: int = 5) -> Dict:
        """
        Complete workflow: analyze input pattern and generate variations.

        Args:
            input_image_path: Path to input kolam image
            num_variations: Number of variations to generate

        Returns:
            Complete analysis and generation results
        """
        logger.info("Starting analyze and generate workflow...")

        # Step 1: Recognize the input pattern
        recognition_results = self.recognize_pattern(input_image_path)

        # Step 2: Generate variations
        generated_patterns = self.generate_kolam_patterns(num_variations)

        # Step 3: Create comprehensive results
        workflow_results = {
            'input_analysis': recognition_results,
            'generated_variations': generated_patterns,
            'num_variations': len(generated_patterns),
            'workflow_timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0'
        }

        logger.info("Workflow completed successfully")
        return workflow_results

    def save_training_history(self, filepath: str = None):
        """
        Save the training history to a JSON file.

        Args:
            filepath: Path to save history (optional)
        """
        if filepath is None:
            filepath = os.path.join(
                self.config['model_dir'], 'training_history.json')

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert numpy arrays and types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_history = convert_to_serializable(self.training_history)

        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        logger.info(f"Training history saved to {filepath}")

    def load_training_history(self, filepath: str = None):
        """
        Load training history from a JSON file.

        Args:
            filepath: Path to load history from (optional)
        """
        if filepath is None:
            filepath = os.path.join(
                self.config['model_dir'], 'training_history.json')

        if not os.path.exists(filepath):
            logger.warning(f"Training history file not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            self.training_history = json.load(f)

        logger.info(f"Training history loaded from {filepath}")

    def create_visualization_report(self, save_dir: str = 'backend/reports'):
        """
        Create a comprehensive visualization report of the pipeline.

        Args:
            save_dir: Directory to save the report
        """
        os.makedirs(save_dir, exist_ok=True)

        # Create summary report
        report = f"""
# Kolam AI Pipeline Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Information
- Total Images: {self.training_history.get('dataset_info', {}).get('total_samples', 'N/A')}
- Number of Classes: {self.training_history.get('dataset_info', {}).get('num_classes', 'N/A')}
- Image Shape: {self.training_history.get('dataset_info', {}).get('image_shape', 'N/A')}

## CNN Model Performance
- Final Training Accuracy: {self.training_history.get('cnn_results', {}).get('final_train_accuracy', 'N/A'):.4f}
- Final Validation Accuracy: {self.training_history.get('cnn_results', {}).get('final_val_accuracy', 'N/A'):.4f}
- Test Accuracy: {self.training_history.get('cnn_results', {}).get('cnn_evaluation', {}).get('test_accuracy', 'N/A'):.4f}

## GAN Model Training
- Epochs Trained: {self.training_history.get('gan_results', {}).get('gan_epochs_trained', 'N/A')}
- Final Discriminator Loss: {self.training_history.get('gan_results', {}).get('final_d_loss', 'N/A'):.4f}
- Final Generator Loss: {self.training_history.get('gan_results', {}).get('final_g_loss', 'N/A'):.4f}

## Model Files
- CNN Model: {os.path.join(self.config['model_dir'], 'kolam_cnn_final.h5')}
- GAN Generator: {os.path.join(self.config['model_dir'], 'kolam_generator.h5')}
- GAN Discriminator: {os.path.join(self.config['model_dir'], 'kolam_discriminator.h5')}

## Usage
The pipeline is ready for:
1. Pattern Recognition: Use recognize_pattern() method
2. Pattern Generation: Use generate_kolam_patterns() method
3. Complete Workflow: Use analyze_and_generate() method
        """

        # Save report
        report_path = os.path.join(save_dir, 'pipeline_report.md')
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Visualization report saved to {report_path}")


def main():
    """
    Main function to demonstrate the complete AI pipeline.
    """
    # Initialize pipeline
    pipeline = KolamAIPipeline(img_height=128, img_width=128)

    # Train the complete pipeline (reduced epochs for demo)
    dataset_path = 'backend/rangoli_dataset_complete/images'
    metadata_path = 'backend/rangoli_dataset_complete/metadata'

    training_results = pipeline.train_full_pipeline(
        dataset_path,
        metadata_path,
        cnn_epochs=10,  # Reduced for demo
        gan_epochs=50   # Reduced for demo
    )

    # Create visualization report
    pipeline.create_visualization_report()

    # Demonstrate usage
    print("\n" + "="*60)
    print("AI PIPELINE DEMONSTRATION")
    print("="*60)

    # Generate some sample patterns
    print("Generating sample kolam patterns...")
    patterns = pipeline.generate_kolam_patterns(3)

    for i, pattern in enumerate(patterns):
        print(f"Generated pattern {i+1}: {pattern['pattern_id']}")

    # Create report
    print("\nPipeline report created successfully!")
    print("Check backend/reports/pipeline_report.md for detailed information")


if __name__ == "__main__":
    main()
