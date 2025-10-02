"""
Test Script for Kolam AI Models

This script provides a simple way to test the implemented CNN and GAN models
with the available dataset.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, List
import logging

# Import our custom modules
try:
    from kolam_cnn_model import KolamCNNModel
    from kolam_gan_model import KolamGAN
    from kolam_data_pipeline import KolamDataPipeline
    from kolam_ai_pipeline import KolamAIPipeline
    from kolam_model_evaluation import KolamModelEvaluator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all model files are created successfully")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_pipeline():
    """
    Test the data pipeline functionality.
    """
    print("\n" + "="*50)
    print("TESTING DATA PIPELINE")
    print("="*50)

    try:
        # Initialize pipeline
        # Smaller for faster testing
        pipeline = KolamDataPipeline(img_height=64, img_width=64)

        # Load dataset
        dataset_path = 'backend/rangoli_dataset_complete/images'
        metadata_path = 'backend/rangoli_dataset_complete/metadata'

        print("Loading dataset...")
        dataset_info = pipeline.load_dataset(dataset_path, metadata_path)
        print(
            f"✓ Dataset loaded: {dataset_info['total_images']} images, {dataset_info['num_classes']} classes")

        # Analyze dataset
        print("Analyzing dataset...")
        analysis = pipeline.analyze_dataset()
        print(
            f"✓ Dataset analyzed. Imbalance ratio: {analysis['imbalance_ratio']:.2f}")

        # Create visualizations
        print("Creating dataset visualizations...")
        pipeline.visualize_dataset()
        print("✓ Visualizations created")

        # Prepare for CNN
        print("Preparing data for CNN...")
        cnn_data = pipeline.prepare_for_cnn(batch_size=16)
        print(
            f"✓ CNN data prepared: {cnn_data['num_train_batches']} train batches")

        # Prepare for GAN
        print("Preparing data for GAN...")
        gan_data = pipeline.prepare_for_gan()
        print(f"✓ GAN data prepared: {gan_data.shape}")

        return {
            'pipeline': pipeline,
            'dataset_info': dataset_info,
            'cnn_data': cnn_data,
            'gan_data': gan_data
        }

    except Exception as e:
        print(f"✗ Data pipeline test failed: {str(e)}")
        return None


def test_cnn_model(data_info: Dict):
    """
    Test the CNN model functionality.
    """
    print("\n" + "="*50)
    print("TESTING CNN MODEL")
    print("="*50)

    try:
        # Initialize model
        cnn_model = KolamCNNModel(img_height=64, img_width=64)

        # Get data
        X_train = data_info['cnn_data']['splits']['X_train']
        y_train = data_info['cnn_data']['splits']['y_train']
        X_test = data_info['cnn_data']['splits']['X_test']
        y_test = data_info['cnn_data']['splits']['y_test']

        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(
            y_train, data_info['pipeline'].num_classes)
        y_test_cat = tf.keras.utils.to_categorical(
            y_test, data_info['pipeline'].num_classes)

        print("Building CNN model...")
        model = cnn_model.build_model()
        print(f"✓ CNN model built: {model.count_params()} parameters")

        print("Training CNN model (5 epochs for testing)...")
        history = cnn_model.train_model(
            X_train, y_train_cat, X_test, y_test_cat)

        print("Evaluating CNN model...")
        results = cnn_model.evaluate_model(X_test, y_test_cat)
        print(
            f"✓ CNN evaluation completed. Accuracy: {results['test_accuracy']:.4f}")

        # Plot training history
        print("Creating training plots...")
        cnn_model.plot_training_history()
        print("✓ Training plots created")

        return {
            'model': cnn_model,
            'results': results,
            'history': history
        }

    except Exception as e:
        print(f"✗ CNN model test failed: {str(e)}")
        return None


def test_gan_model(data_info: Dict):
    """
    Test the GAN model functionality.
    """
    print("\n" + "="*50)
    print("TESTING GAN MODEL")
    print("="*50)

    try:
        # Initialize GAN
        # Smaller for faster testing
        gan_model = KolamGAN(img_height=32, img_width=32)

        # Get GAN data
        gan_data = data_info['gan_data']

        print("Building GAN model...")
        gan_model.build_generator()
        gan_model.build_discriminator()
        gan_model.build_gan()
        print("✓ GAN model built")

        print("Training GAN model (10 epochs for testing)...")
        # Use subset for faster testing
        subset_size = min(100, len(gan_data))
        test_data = gan_data[:subset_size]

        # Create temporary directory for testing
        temp_dir = 'backend/temp_test_gan'
        os.makedirs(temp_dir, exist_ok=True)

        # Save test images
        for i in range(len(test_data)):
            img = (test_data[i] * 127.5 + 127.5).astype('uint8')
            img_pil = tf.keras.preprocessing.image.array_to_img(img)
            img_pil.save(f'{temp_dir}/test_{i}.jpg')

        # Train for fewer epochs
        gan_model.train(temp_dir, epochs=5)

        print("Generating sample images...")
        sample_images = gan_model.generate_kolam(4)

        if sample_images is not None:
            print(f"✓ Generated {len(sample_images)} sample images")

            # Save sample images
            os.makedirs('backend/generated_samples', exist_ok=True)
            for i, img in enumerate(sample_images):
                img_pil = tf.keras.preprocessing.image.array_to_img(
                    (img * 255).astype('uint8'))
                img_pil.save(f'backend/generated_samples/sample_{i}.png')

            print("✓ Sample images saved")

        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

        return {
            'model': gan_model,
            'sample_images': sample_images
        }

    except Exception as e:
        print(f"✗ GAN model test failed: {str(e)}")
        return None


def test_ai_pipeline():
    """
    Test the integrated AI pipeline.
    """
    print("\n" + "="*50)
    print("TESTING AI PIPELINE")
    print("="*50)

    try:
        # Initialize pipeline
        ai_pipeline = KolamAIPipeline(img_height=32, img_width=32)

        print("Testing pipeline initialization...")
        print("✓ AI pipeline initialized")

        # Test dataset preparation
        print("Testing dataset preparation...")
        dataset_path = 'backend/rangoli_dataset_complete/images'
        metadata_path = 'backend/rangoli_dataset_complete/metadata'

        prep_info = ai_pipeline.prepare_dataset(dataset_path, metadata_path)
        print(f"✓ Dataset prepared: {prep_info['total_samples']} samples")

        return {
            'pipeline': ai_pipeline,
            'prep_info': prep_info
        }

    except Exception as e:
        print(f"✗ AI pipeline test failed: {str(e)}")
        return None


def main():
    """
    Main test function.
    """
    print("KOLAM AI MODELS TEST SUITE")
    print("=" * 50)
    print("This script tests all implemented models with the dataset")

    # Test data pipeline
    data_info = test_data_pipeline()
    if data_info is None:
        print("Data pipeline test failed. Exiting.")
        return

    # Test CNN model
    cnn_info = test_cnn_model(data_info)
    if cnn_info is None:
        print("CNN model test failed. Continuing with other tests.")

    # Test GAN model
    gan_info = test_gan_model(data_info)
    if gan_info is None:
        print("GAN model test failed. Continuing with other tests.")

    # Test AI pipeline
    pipeline_info = test_ai_pipeline()
    if pipeline_info is None:
        print("AI pipeline test failed.")

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    tests_passed = 0
    total_tests = 4

    if data_info:
        print("✓ Data Pipeline: PASSED")
        tests_passed += 1
    else:
        print("✗ Data Pipeline: FAILED")

    if cnn_info:
        print("✓ CNN Model: PASSED")
        tests_passed += 1
    else:
        print("✗ CNN Model: FAILED")

    if gan_info:
        print("✓ GAN Model: PASSED")
        tests_passed += 1
    else:
        print("✗ GAN Model: FAILED")

    if pipeline_info:
        print("✓ AI Pipeline: PASSED")
        tests_passed += 1
    else:
        print("✗ AI Pipeline: FAILED")

    print(f"\nTests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Models are ready for use.")
    elif tests_passed >= 2:
        print("⚠️  Some tests passed. Models may need debugging.")
    else:
        print("❌ Most tests failed. Please check the implementation.")

    print("\nNext steps:")
    print("1. Review any error messages above")
    print("2. Check that all dependencies are installed")
    print("3. Verify dataset path and structure")
    print("4. Run individual model training with full epochs")
    print("5. Create evaluation reports")


if __name__ == "__main__":
    main()
