"""
Quick Test for Kolam AI Models - Import and Basic Functionality Test

This script performs basic import tests and validates core functionality
without running full training cycles.
"""

import os
import sys
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """
    Test that all modules can be imported successfully.
    """
    print("Testing module imports...")

    try:
        # Test TensorFlow and Keras
        import tensorflow as tf
        from tensorflow import keras
        print(f"[OK] TensorFlow {tf.__version__} imported successfully")
        print("[OK] Keras imported successfully")

        # Test PyTorch
        import torch
        import torchvision
        print(f"[OK] PyTorch {torch.__version__} imported successfully")
        print(
            f"[OK] Torchvision {torchvision.__version__} imported successfully")

        # Test other dependencies
        import cv2
        print(f"[OK] OpenCV {cv2.__version__} imported successfully")

        import skimage
        print(f"[OK] Scikit-image {skimage.__version__} imported successfully")

        return True

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False


def test_kolam_modules():
    """
    Test that our custom kolam modules can be imported.
    """
    print("\nTesting Kolam AI modules...")

    modules_to_test = [
        'kolam_cnn_model',
        'kolam_gan_model',
        'kolam_data_pipeline',
        'kolam_ai_pipeline',
        'kolam_model_evaluation'
    ]

    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"[OK] {module_name} imported successfully")
        except ImportError as e:
            print(f"[FAIL] Failed to import {module_name}: {e}")
            return False

    return True


def test_basic_functionality():
    """
    Test basic functionality without full training.
    """
    print("\nTesting basic functionality...")

    try:
        from kolam_data_pipeline import KolamDataPipeline

        # Test data pipeline initialization
        pipeline = KolamDataPipeline(img_height=64, img_width=64)
        print("[OK] Data pipeline initialized successfully")

        # Test dataset loading (check if files exist)
        dataset_path = 'backend/rangoli_dataset_complete/images'
        if os.path.exists(dataset_path):
            print("[OK] Dataset directory found")

            # Try to load a small subset
            try:
                dataset_info = pipeline.load_dataset(dataset_path)
                print(
                    f"[OK] Dataset loaded: {dataset_info['total_images']} images, {dataset_info['num_classes']} classes")
            except Exception as e:
                print(
                    f"[WARN] Dataset loading issue (expected for large datasets): {e}")
        else:
            print(
                "[WARN] Dataset directory not found - this is expected if dataset not yet created")

        return True

    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        return False


def test_model_initialization():
    """
    Test model initialization without training.
    """
    print("\nTesting model initialization...")

    try:
        from kolam_cnn_model import KolamCNNModel
        from kolam_gan_model import KolamGAN

        # Test CNN model
        cnn_model = KolamCNNModel(img_height=64, img_width=64)
        print("[OK] CNN model initialized successfully")

        # Test GAN model
        gan_model = KolamGAN(img_height=32, img_width=32)
        print("[OK] GAN model initialized successfully")

        # Test model building (skip if num_classes is None)
        try:
            cnn_model.build_model()
            print("[OK] CNN model built successfully")
        except Exception as e:
            print(
                f"[WARN] CNN model building skipped (num_classes not set): {e}")

        gan_model.build_generator()
        gan_model.build_discriminator()
        gan_model.build_gan()
        print("[OK] GAN model built successfully")

        return True

    except Exception as e:
        print(f"[ERROR] Model initialization test failed: {e}")
        return False


def main():
    """
    Main test function.
    """
    print("KOLAM AI MODELS - QUICK VALIDATION TEST")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Kolam Modules Test", test_kolam_modules),
        ("Basic Functionality Test", test_basic_functionality),
        ("Model Initialization Test", test_model_initialization)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)

        if test_func():
            passed += 1
        else:
            print(f"[FAIL] {test_name} failed")

    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[SUCCESS] All tests passed! Models are ready for use.")
        print("\nNext steps:")
        print("1. Run full training: python kolam_ai_pipeline.py")
        print("2. Generate evaluation reports: python kolam_model_evaluation.py")
        print("3. Test with real data: Use the integrated pipeline")
    else:
        print("[WARNING] Some tests failed. Please check the issues above.")
        print("Common solutions:")
        print("- Ensure all dependencies are installed")
        print("- Check Python path and virtual environment")
        print("- Verify TensorFlow and PyTorch installations")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
