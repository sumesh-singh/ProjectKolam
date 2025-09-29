"""
Kolam Model Evaluation and Testing

This module provides comprehensive evaluation and testing utilities for both CNN and GAN models,
including performance metrics, hyperparameter tuning, and model comparison.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import pandas as pd
# KerasClassifier is deprecated in newer versions
# from keras.wrappers.scikit_learn import KerasClassifier
import itertools

# Import our custom modules
from kolam_cnn_model import KolamCNNModel
from kolam_gan_model import KolamGAN
from kolam_data_pipeline import KolamDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KolamModelEvaluator:
    """
    Comprehensive evaluation system for kolam AI models.
    """

    def __init__(self, cnn_model: KolamCNNModel = None, gan_model: KolamGAN = None):
        """
        Initialize the evaluator.

        Args:
            cnn_model: CNN model instance
            gan_model: GAN model instance
        """
        self.cnn_model = cnn_model
        self.gan_model = gan_model
        self.evaluation_results = {}
        self.best_params = {}

    def evaluate_cnn_model(self, X_test: np.ndarray, y_test: np.ndarray,
                           model_path: str = None) -> Dict:
        """
        Comprehensive evaluation of the CNN model.

        Args:
            X_test: Test images
            y_test: Test labels
            model_path: Path to saved model (optional)

        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting comprehensive CNN model evaluation...")

        # Load model if path provided
        if model_path and self.cnn_model is None:
            self.cnn_model = KolamCNNModel()
            self.cnn_model.load_model(model_path)

        if self.cnn_model is None:
            raise ValueError("No CNN model provided")

        # Get predictions
        y_pred = self.cnn_model.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Basic metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(
            y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(
            y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

        # Per-class metrics
        class_report = classification_report(
            y_true_classes, y_pred_classes,
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # ROC AUC (for multiclass)
        try:
            # One-hot encode for ROC AUC
            y_test_bin = y_test
            if y_test_bin.shape[1] == 1:
                y_test_bin = tf.keras.utils.to_categorical(y_true_classes)

            roc_auc = roc_auc_score(
                y_test_bin, y_pred, multi_class='ovr', average='weighted')
        except:
            roc_auc = None

        # Model complexity metrics
        total_params = self.cnn_model.model.count_params()
        trainable_params = np.sum([tf.keras.backend.count_params(
            w) for w in self.cnn_model.model.trainable_weights])

        # Inference time test
        import time
        start_time = time.time()
        for _ in range(100):  # Test on 100 samples
            _ = self.cnn_model.model.predict(X_test[:1], verbose=0)
        avg_inference_time = (time.time() - start_time) / 100

        evaluation_results = {
            'basic_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc else None
            },
            'per_class_metrics': class_report,
            'confusion_matrix': {
                'raw': cm.tolist(),
                'normalized': cm_normalized.tolist()
            },
            'model_info': {
                'total_parameters': int(total_params),
                'trainable_parameters': int(trainable_params),
                'avg_inference_time_ms': float(avg_inference_time * 1000)
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }

        self.evaluation_results['cnn'] = evaluation_results
        logger.info(f"CNN evaluation completed. Accuracy: {accuracy:.4f}")
        return evaluation_results

    def evaluate_gan_model(self, real_images: np.ndarray, num_samples: int = 1000) -> Dict:
        """
        Evaluate GAN model quality and performance.

        Args:
            real_images: Real images for comparison
            num_samples: Number of samples to generate for evaluation

        Returns:
            GAN evaluation results
        """
        logger.info("Starting GAN model evaluation...")

        if self.gan_model is None:
            raise ValueError("No GAN model provided")

        # Generate samples
        noise = np.random.normal(
            0, 1, (num_samples, self.gan_model.latent_dim))
        generated_images = self.gan_model.generator.predict(noise, verbose=0)

        # Convert generated images from [-1, 1] to [0, 1] for comparison
        generated_images = 0.5 * generated_images + 0.5

        # Diversity metrics (how different are generated images)
        diversity_score = self._calculate_diversity_score(generated_images)

        # Quality metrics (how close to real images)
        quality_score = self._calculate_quality_score(
            generated_images, real_images)

        # Discriminator evaluation on generated vs real
        real_subset = real_images[:min(num_samples, len(real_images))]
        real_labels = np.ones((len(real_subset), 1))
        fake_labels = np.zeros((num_samples, 1))

        real_scores = self.gan_model.discriminator.predict(
            real_subset, verbose=0)
        fake_scores = self.gan_model.discriminator.predict(
            generated_images, verbose=0)

        discriminator_accuracy = np.mean(
            (real_scores > 0.5).astype(int)
        ) * 0.5 + np.mean(
            (fake_scores < 0.5).astype(int)
        ) * 0.5

        # Training stability (based on loss progression)
        stability_score = self._calculate_training_stability()

        evaluation_results = {
            'generation_metrics': {
                'diversity_score': float(diversity_score),
                'quality_score': float(quality_score),
                'discriminator_accuracy': float(discriminator_accuracy),
                'stability_score': float(stability_score)
            },
            'sample_statistics': {
                'generated_mean': float(np.mean(generated_images)),
                'generated_std': float(np.std(generated_images)),
                'real_mean': float(np.mean(real_images)),
                'real_std': float(np.std(real_images))
            },
            'model_info': {
                'latent_dimension': self.gan_model.latent_dim,
                'generator_params': self.gan_model.generator.count_params(),
                'discriminator_params': self.gan_model.discriminator.count_params()
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }

        self.evaluation_results['gan'] = evaluation_results
        logger.info("GAN evaluation completed")
        return evaluation_results

    def _calculate_diversity_score(self, generated_images: np.ndarray) -> float:
        """
        Calculate diversity score based on pairwise image differences.

        Args:
            generated_images: Generated image samples

        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        # Calculate pairwise structural similarity or simple pixel differences
        # Use subset for efficiency
        num_samples = min(100, len(generated_images))
        subset = generated_images[:num_samples]

        # Simple diversity metric based on variance
        pixel_variance = np.var(subset)
        normalized_variance = 1.0 / \
            (1.0 + np.exp(-pixel_variance / 0.01))  # Sigmoid normalization

        return float(min(normalized_variance, 1.0))

    def _calculate_quality_score(self, generated_images: np.ndarray, real_images: np.ndarray) -> float:
        """
        Calculate quality score based on similarity to real images.

        Args:
            generated_images: Generated images
            real_images: Real images for comparison

        Returns:
            Quality score (0-1, higher is better quality)
        """
        # Simple quality metric based on pixel value distribution similarity
        gen_mean, gen_std = np.mean(generated_images), np.std(generated_images)
        real_mean, real_std = np.mean(real_images), np.std(real_images)

        # Calculate similarity in distribution
        mean_diff = abs(gen_mean - real_mean)
        std_diff = abs(gen_std - real_std)

        quality_score = 1.0 / (1.0 + mean_diff + std_diff)

        return float(quality_score)

    def _calculate_training_stability(self) -> float:
        """
        Calculate training stability based on loss progression.

        Returns:
            Stability score (0-1, higher is more stable)
        """
        if not hasattr(self.gan_model, 'd_losses') or not self.gan_model.d_losses:
            return 0.5  # Default stability if no training history

        d_losses = np.array(self.gan_model.d_losses[-100:])  # Last 100 steps
        g_losses = np.array(self.gan_model.g_losses[-100:])

        # Stability based on loss variance (lower variance = more stable)
        d_stability = 1.0 / (1.0 + np.var(d_losses))
        g_stability = 1.0 / (1.0 + np.var(g_losses))

        overall_stability = (d_stability + g_stability) / 2.0

        return float(overall_stability)

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              param_grid: Dict = None) -> Dict:
        """
        Perform hyperparameter tuning for the CNN model.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            param_grid: Parameter grid for tuning

        Returns:
            Tuning results
        """
        logger.info("Starting hyperparameter tuning...")

        if param_grid is None:
            param_grid = {
                'learning_rate': [0.001, 0.0001],
                'batch_size': [16, 32, 64],
                'dense_units': [256, 512, 1024]
            }

        # Convert labels to categorical for Keras
        y_train_cat = tf.keras.utils.to_categorical(y_train)
        y_val_cat = tf.keras.utils.to_categorical(y_val)

        best_accuracy = 0
        best_params = {}

        # Grid search
        for params in self._param_combinations(param_grid):
            logger.info(f"Testing parameters: {params}")

            # Create model with current parameters
            model = self._create_model_for_tuning(
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                dense_units=params.get('dense_units', 512)
            )

            # Train model
            history = model.fit(
                X_train, y_train_cat,
                batch_size=params['batch_size'],
                epochs=10,  # Short training for tuning
                validation_data=(X_val, y_val_cat),
                verbose=0
            )

            # Evaluate
            val_accuracy = history.history['val_accuracy'][-1]

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = params

        tuning_results = {
            'best_parameters': best_params,
            'best_accuracy': float(best_accuracy),
            'parameter_grid': param_grid,
            'tuning_timestamp': datetime.now().isoformat()
        }

        self.best_params = best_params
        logger.info(
            f"Hyperparameter tuning completed. Best accuracy: {best_accuracy:.4f}")
        return tuning_results

    def _param_combinations(self, param_grid: Dict) -> List[Dict]:
        """
        Generate all combinations of parameters.

        Args:
            param_grid: Dictionary of parameter lists

        Returns:
            List of parameter combinations
        """
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)

        return param_combinations

    def _create_model_for_tuning(self, learning_rate: float = 0.001,
                                 batch_size: int = 32, dense_units: int = 512) -> keras.Model:
        """
        Create CNN model for hyperparameter tuning.

        Args:
            learning_rate: Learning rate
            batch_size: Batch size
            dense_units: Dense layer units

        Returns:
            Keras model
        """
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(dense_units, activation='relu'),
            keras.layers.Dense(self.cnn_model.num_classes,
                               activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_evaluation_report(self, save_path: str = 'backend/evaluation_reports'):
        """
        Create comprehensive evaluation report.

        Args:
            save_path: Path to save the report
        """
        os.makedirs(save_path, exist_ok=True)

        # Create plots and visualizations
        self._create_evaluation_plots(save_path)

        # Generate text report
        report = self._generate_text_report()

        # Save report
        report_path = os.path.join(save_path, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)

        # Save results as JSON
        results_path = os.path.join(save_path, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {save_path}")

    def _create_evaluation_plots(self, save_path: str):
        """
        Create evaluation plots and visualizations.

        Args:
            save_path: Path to save plots
        """
        # CNN Confusion Matrix
        if 'cnn' in self.evaluation_results:
            cnn_results = self.evaluation_results['cnn']
            cm = np.array(cnn_results['confusion_matrix']['raw'])

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=range(len(cm)),
                        yticklabels=range(len(cm)))
            plt.title('CNN Model - Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(os.path.join(
                save_path, 'cnn_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # GAN Training History
        if 'gan' in self.evaluation_results and hasattr(self.gan_model, 'd_losses'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.gan_model.d_losses,
                     label='Discriminator Loss', alpha=0.7)
            plt.plot(self.gan_model.g_losses,
                     label='Generator Loss', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('GAN Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(
                save_path, 'gan_training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _generate_text_report(self) -> str:
        """
        Generate comprehensive text report.

        Returns:
            Report as string
        """
        report = f"""
# Kolam AI Models Evaluation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## CNN Model Evaluation

### Basic Metrics
"""

        if 'cnn' in self.evaluation_results:
            cnn_metrics = self.evaluation_results['cnn']['basic_metrics']
            roc_auc_str = f"{cnn_metrics['roc_auc']:.4f}" if cnn_metrics['roc_auc'] else 'N/A'
            report += f"""
- **Accuracy**: {cnn_metrics['accuracy']:.4f}
- **Precision**: {cnn_metrics['precision']:.4f}
- **Recall**: {cnn_metrics['recall']:.4f}
- **F1-Score**: {cnn_metrics['f1_score']:.4f}
- **ROC AUC**: {roc_auc_str}
"""

            model_info = self.evaluation_results['cnn']['model_info']
            report += f"""
### Model Information
- **Total Parameters**: {model_info['total_parameters']:,}
- **Trainable Parameters**: {model_info['trainable_parameters']:,}
- **Average Inference Time**: {model_info['avg_inference_time_ms']:.2f} ms
"""

        report += """

## GAN Model Evaluation

"""

        if 'gan' in self.evaluation_results:
            gan_metrics = self.evaluation_results['gan']['generation_metrics']
            report += f"""
### Generation Metrics
- **Diversity Score**: {gan_metrics['diversity_score']:.4f}
- **Quality Score**: {gan_metrics['quality_score']:.4f}
- **Discriminator Accuracy**: {gan_metrics['discriminator_accuracy']:.4f}
- **Stability Score**: {gan_metrics['stability_score']:.4f}
"""

            gan_info = self.evaluation_results['gan']['model_info']
            report += f"""
### Model Information
- **Latent Dimension**: {gan_info['latent_dimension']}
- **Generator Parameters**: {gan_info['generator_params']:,}
- **Discriminator Parameters**: {gan_info['discriminator_params']:,}
"""

        report += """

## Recommendations

### CNN Model Improvements
- Consider data augmentation if accuracy is low
- Try different architectures (ResNet, EfficientNet) for better performance
- Implement learning rate scheduling
- Use ensemble methods for improved accuracy

### GAN Model Improvements
- Increase training epochs for better quality
- Try different architectures (DCGAN, StyleGAN)
- Implement progressive growing
- Add conditional generation based on kolam types

### General Recommendations
- Collect more diverse training data
- Implement cross-validation
- Consider transfer learning for CNN
- Add attention mechanisms for better feature extraction

## Files Generated
- Confusion Matrix: `cnn_confusion_matrix.png`
- GAN Training History: `gan_training_history.png`
- Detailed Results: `evaluation_results.json`
        """

        return report


def main():
    """
    Main function to demonstrate model evaluation.
    """
    print("Kolam Model Evaluation System")
    print("=" * 40)

    # Initialize evaluator
    evaluator = KolamModelEvaluator()

    # Example usage (assuming models are trained)
    print("Evaluation system initialized.")
    print("Load trained models to perform evaluation:")
    print("1. evaluator.cnn_model = KolamCNNModel()")
    print("2. evaluator.gan_model = KolamGAN()")
    print("3. results = evaluator.evaluate_cnn_model(X_test, y_test)")
    print("4. evaluator.create_evaluation_report()")


if __name__ == "__main__":
    main()
