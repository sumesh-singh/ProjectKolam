"""
Enhanced Kolam Training Pipeline with Computer Vision

This module demonstrates how to use the enhanced computer vision preprocessing
techniques with the improved CNN model for superior kolam pattern recognition.
"""

from kolam_data_pipeline import KolamDataPipeline
from kolam_cnn_model import KolamCNNModel
from kolam_cv_enhancement import KolamCVEnhancement
import os
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow as tf
-  # Add backend to path for imports
-sys.path.append('backend')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
sys.path.append('backend')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedKolamTrainingPipeline:
    """
    Complete training pipeline with computer vision enhancements.
    """

    def __init__(self, img_height: int = 224, img_width: int = 224):
        """
        Initialize the enhanced training pipeline.

        Args:
            img_height: Image height
            img_width: Image width
        """
        self.img_height = img_height
        self.img_width = img_width

        # Initialize components
        self.cv_enhancer = KolamCVEnhancement()
        self.data_pipeline = KolamDataPipeline(img_height, img_width)
        self.cnn_model = KolamCNNModel(img_height, img_width)

        # Configuration
        self.config = {
            'use_cv_enhancement': True,
            'multi_channel_input': True,
            'batch_size': 16,
            'epochs': 30,
            'learning_rate': 0.001,
            'save_enhanced_images': True,
            'output_dir': 'backend/enhanced_training'
        }

        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def prepare_enhanced_dataset(self, dataset_path: str, sample_size: int = None) -> dict:
        """
        Prepare dataset with computer vision enhancements.

        Args:
            dataset_path: Path to the dataset
            sample_size: Optional sample size limit for testing

        Returns:
            Dictionary with enhanced dataset information
        """
        logger.info(
            "Preparing enhanced dataset with computer vision preprocessing...")

        # Load basic dataset
        dataset_info = self.data_pipeline.load_dataset(dataset_path)

        if sample_size and len(self.data_pipeline.images) > sample_size:
            # Use subset for faster testing
            indices = np.random.choice(
                len(self.data_pipeline.images), sample_size, replace=False
            )
            self.data_pipeline.images = self.data_pipeline.images[indices]
            self.data_pipeline.labels = self.data_pipeline.labels[indices]
            logger.info(f"Using sample of {sample_size} images for testing")

        # Apply computer vision enhancements
        enhanced_images = []
        cv_features_list = []

        logger.info("Applying computer vision enhancements to dataset...")

        for i, img in enumerate(self.data_pipeline.images):
            # Validate and convert image range
            if img.dtype == np.uint8:
                img_uint8 = img
            elif img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                logger.warning(
                    f"Unexpected image range for image {i}, skipping conversion")
                img_uint8 = img.astype(np.uint8)

            # Apply CV preprocessing
            try:
                cv_results = self.cv_enhancer.preprocess_image_from_array(
                    img_uint8)
                # Store enhanced image and features
                enhanced_images.append(cv_results['combined'])
                cv_features_list.append(cv_results['features'])
            except Exception as e:
                logger.error(f"Failed to enhance image {i}: {e}")
                continue

            if (i + 1) % 50 == 0:
                logger.info(
                    f"Enhanced {i + 1}/{len(self.data_pipeline.images)} images")

        # Update data pipeline with enhanced images
        self.enhanced_pipeline = KolamDataPipeline(
            self.img_height, self.img_width)
        self.enhanced_pipeline.images = np.array(enhanced_images)
        self.enhanced_pipeline.labels = self.data_pipeline.labels[:len(
            enhanced_images)]
        self.enhanced_pipeline.num_classes = self.data_pipeline.num_classes

        # Split enhanced dataset
        splits = self.enhanced_pipeline.split_dataset()

        # Create TensorFlow datasets
        train_dataset = self.enhanced_pipeline.create_tf_datasets(
            splits['X_train'], splits['y_train'],
            batch_size=self.config['batch_size'], shuffle=True
        )

        val_dataset = self.enhanced_pipeline.create_tf_datasets(
            splits['X_val'], splits['y_val'],
            batch_size=self.config['batch_size'], shuffle=False
        )

        test_dataset = self.enhanced_pipeline.create_tf_datasets(
            splits['X_test'], splits['y_test'],
            batch_size=self.config['batch_size'], shuffle=False
        )

        preparation_info = {
            'dataset_info': dataset_info,
            'cv_features': cv_features_list,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'splits': splits,
            'num_enhanced_channels': enhanced_images[0].shape[-1] if enhanced_images else 3
        }

        logger.info(
            f"Enhanced dataset prepared: {len(enhanced_images)} images")
        logger.info(
            f"Enhanced channels: {preparation_info['num_enhanced_channels']}")

        return preparation_info

    def train_enhanced_model(self, dataset_path: str, sample_size: int = 200) -> dict:
        """
        Train the CNN model with enhanced computer vision preprocessing.

        Args:
            dataset_path: Path to the dataset
            sample_size: Sample size for testing (None for full dataset)

        Returns:
            Training results dictionary
        """
        logger.info("Starting enhanced model training...")

        # Step 1: Prepare enhanced dataset
        prep_info = self.prepare_enhanced_dataset(dataset_path, sample_size)

        # Step 2: Update CNN model configuration for multi-channel input
        self.cnn_model.num_classes = self.enhanced_pipeline.num_classes
        self.cnn_model.config.update({
            'use_multi_channel': True,
            'num_input_channels': prep_info['num_enhanced_channels']
        })

        # Step 3: Build enhanced model
        enhanced_model = self.cnn_model.build_enhanced_model()
        self.cnn_model.model = enhanced_model

        # Step 4: Extract training data
        X_train = prep_info['splits']['X_train']
        y_train = prep_info['splits']['y_train']
        X_val = prep_info['splits']['X_val']
        y_val = prep_info['splits']['y_val']

        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(
            y_train, self.enhanced_pipeline.num_classes
        )
        y_val_cat = tf.keras.utils.to_categorical(
            y_val, self.enhanced_pipeline.num_classes
        )

        # Step 5: Train the enhanced model
        history = self.cnn_model.train_model(
            X_train, y_train_cat, X_val, y_val_cat
        )

        # Step 6: Evaluate the model
        y_test_cat = tf.keras.utils.to_categorical(
            prep_info['splits']['y_test'],
            self.enhanced_pipeline.num_classes
        )
        results = self.cnn_model.evaluate_model(
            prep_info['splits']['X_test'],
            y_test_cat
        )

        # Step 7: Save enhanced model
        model_path = os.path.join(
            self.config['output_dir'], 'enhanced_kolam_cnn.h5'
        )
        self.cnn_model.save_model(model_path)

        # Step 8: Create training report
        training_results = {
            'model_type': 'enhanced_multi_channel',
            'training_history': history.history,
            'evaluation_results': results,
            'cv_enhancement_used': True,
            'num_enhanced_channels': prep_info['num_enhanced_channels'],
            'dataset_info': prep_info['dataset_info'],
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'test_accuracy': results['test_accuracy'],
            'training_timestamp': datetime.now().isoformat()
        }

        # Save training results
        self._save_training_report(training_results)

        logger.info("Enhanced model training completed successfully!")
        logger.info(
            f"Final validation accuracy: {training_results['final_val_accuracy']:.4f}")
        logger.info(f"Test accuracy: {training_results['test_accuracy']:.4f}")

        return training_results

    def demonstrate_cv_enhancement(self, test_image_path: str = None) -> dict:
        """
        Demonstrate computer vision enhancement on a single image.

        Args:
            test_image_path: Path to test image (optional)

        Returns:
            CV enhancement results dictionary
        """
        logger.info("Demonstrating computer vision enhancement...")

        if test_image_path is None:
            # Use one of the existing kolam images
            test_image_path = 'static/mandalaKolam.jpg'

        if not os.path.exists(test_image_path):
            logger.error(f"Test image not found: {test_image_path}")
            return None

        try:
            # Apply computer vision preprocessing
            cv_results = self.cv_enhancer.preprocess_image(test_image_path)

            # Create visualization
            save_path = os.path.join(
                self.config['output_dir'], 'cv_enhancement_demo.png'
            )
            fig = self.cv_enhancer.visualize_preprocessing(
                cv_results,
                save_path=save_path
            )

            # Print enhancement statistics
            features = cv_results['features']
            print("\n" + "="*60)
            print("COMPUTER VISION ENHANCEMENT RESULTS")
            print("="*60)
            print(
                f"Number of pattern components: {features['num_components']}")
            print(f"Pattern symmetry score: {features['symmetry_score']:.3f}")
            print(f"Pattern complexity: {features['complexity']:.3f}")
            print(
                f"Mean edge intensity: {features['mean_edge_intensity']:.3f}")
            print(f"Pattern density: {features['pattern_density']:.3f}")
            print(f"Visualization saved to: {save_path}")

            return cv_results

        except Exception as e:
            logger.error(f"CV preprocessing failed: {e}")
            return None

    def _save_training_report(self, results: dict):
        """
        Save training report to file.

        Args:
            results: Training results dictionary
        """
        report_path = os.path.join(
            self.config['output_dir'], 'enhanced_training_report.txt'
        )

        try:
            with open(report_path, 'w') as f:
                f.write("ENHANCED KOLAM PATTERN RECOGNITION TRAINING REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated on: {results['training_timestamp']}\n\n")
                f.write(f"Model Type: {results['model_type']}\n")
                f.write(
                    f"Computer Vision Enhancement: {results['cv_enhancement_used']}\n")
                f.write(
                    f"Multi-channel Input: {results['num_enhanced_channels']} channels\n\n")
                f.write("TRAINING RESULTS:\n")
                f.write(
                    f"- Final Training Accuracy: {results['final_train_accuracy']:.4f}\n")
                f.write(
                    f"- Final Validation Accuracy: {results['final_val_accuracy']:.4f}\n")
                f.write(f"- Test Accuracy: {results['test_accuracy']:.4f}\n\n")
                f.write(
                    f"Dataset: {results['dataset_info']['total_images']} images\n")
                f.write(f"Classes: {results['dataset_info']['num_classes']}\n")

            logger.info(f"Training report saved to {report_path}")

            # Also save as JSON for programmatic access
            json_path = report_path.replace('.txt', '.json')
            with open(json_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in results.items()
                }
                json.dump(json_results, f, indent=2, default=str)
            logger.info(f"Training results saved to {json_path}")

        except IOError as e:
            logger.error(f"Failed to save training report: {e}")

    def compare_models(self, dataset_path: str, sample_size: int = 100):
        """
        Compare enhanced model with standard model.

        Args:
            dataset_path: Path to dataset
            sample_size: Number of samples to use
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON: Standard vs Enhanced")
        print("="*60)

        # Train enhanced model
        print("\nTraining Enhanced Model...")
        enhanced_results = self.train_enhanced_model(dataset_path, sample_size)

        # Train standard model (without CV enhancement)
        print("\nTraining Standard Model...")
        self.cnn_model.config['use_multi_channel'] = False
        self.cnn_model.config['num_input_channels'] = 3

        # Reload dataset without enhancement
        dataset_info = self.data_pipeline.load_dataset(dataset_path)
        if sample_size and len(self.data_pipeline.images) > sample_size:
            indices = np.random.choice(
                len(self.data_pipeline.images), sample_size, replace=False
            )
            self.data_pipeline.images = self.data_pipeline.images[indices]
            self.data_pipeline.labels = self.data_pipeline.labels[indices]

        splits = self.data_pipeline.split_dataset()

        # Build and train standard model
        self.cnn_model.num_classes = self.data_pipeline.num_classes
        standard_model = self.cnn_model.build_model()
        self.cnn_model.model = standard_model

        y_train_cat = tf.keras.utils.to_categorical(
            splits['y_train'], self.data_pipeline.num_classes
        )
        y_val_cat = tf.keras.utils.to_categorical(
            splits['y_val'], self.data_pipeline.num_classes
        )

        history = self.cnn_model.train_model(
            splits['X_train'], y_train_cat,
            splits['X_val'], y_val_cat
        )

        y_test_cat = tf.keras.utils.to_categorical(
            splits['y_test'], self.data_pipeline.num_classes
        )
        standard_results = self.cnn_model.evaluate_model(
            splits['X_test'], y_test_cat
        )

        # Compare results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(
            f"Standard Model Accuracy: {standard_results['test_accuracy']:.4f}")
        print(
            f"Enhanced Model Accuracy: {enhanced_results['test_accuracy']:.4f}")
        improvement = (
            enhanced_results['test_accuracy'] - standard_results['test_accuracy']) * 100
        print(f"Improvement: {improvement:.2f}%")
        print("="*60)


def main():
    """
    Main function to demonstrate the enhanced training pipeline.
    """
    print("ENHANCED KOLAM PATTERN RECOGNITION SYSTEM")
    print("="*60)

    # Initialize enhanced pipeline
    pipeline = EnhancedKolamTrainingPipeline(img_height=128, img_width=128)

    # Step 1: Demonstrate computer vision enhancement
    print("\n1. COMPUTER VISION ENHANCEMENT DEMONSTRATION")
    print("-" * 50)
    cv_results = pipeline.demonstrate_cv_enhancement()

    if cv_results is None:
        print("Skipping training due to missing test image")
        print("Please ensure test images are available in 'static' folder")
        return

    # Step 2: Train enhanced model (using small sample for demo)
    print("\n2. ENHANCED MODEL TRAINING")
    print("-" * 50)

    # Use a small dataset for demonstration
    dataset_path = 'static'  # Use static folder with kolam images

    if os.path.exists(dataset_path):
        try:
            training_results = pipeline.train_enhanced_model(
                dataset_path,
                sample_size=50  # Small sample for demo
            )

            print("\n3. TRAINING SUMMARY")
            print("-" * 50)
            print(
                f"Enhanced Model Accuracy: {training_results['test_accuracy']:.4f}")
            print(
                f"Computer Vision Features: {training_results['num_enhanced_channels']} channels")
            print(
                f"Training Report: {pipeline.config['output_dir']}/enhanced_training_report.txt")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            print(f"Training failed: {str(e)}")
    else:
        print(f"Dataset path not found: {dataset_path}")
        print("Enhanced computer vision module is ready for use!")

    print("\n" + "="*60)
    print("ENHANCED KOLAM RECOGNITION SYSTEM READY!")
    print("="*60)
    print("Features implemented:")
    print("✓ Advanced edge detection")
    print("✓ Multi-stage noise cancellation")
    print("✓ Pattern enhancement algorithms")
    print("✓ Multi-channel CNN input")
    print("✓ Feature extraction and analysis")
    print("✓ Symmetry detection")
    print("✓ Structural pattern recognition")


if __name__ == "__main__":
    main()
