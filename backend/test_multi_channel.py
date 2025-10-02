#!/usr/bin/env python3
"""
Test script for multi-channel CNN implementation.
This script tests the 6-channel preprocessing pipeline to ensure shape consistency.
"""

from kolam_data_pipeline import KolamDataPipeline
from kolam_cnn_model import KolamCNNModel
import os
import sys
import numpy as np
import logging

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_channel_preprocessing():
    """Test the multi-channel preprocessing functionality."""
    logger.info("Testing multi-channel preprocessing...")

    # Initialize CNN model with multi-channel configuration
    cnn_model = KolamCNNModel(img_height=128, img_width=128)

    # Test 1: Check configuration
    logger.info(f"CNN Model Config: {cnn_model.config}")
    assert cnn_model.config['use_multi_channel'] == True
    assert cnn_model.config['num_input_channels'] == 6
    logger.info("✓ Configuration test passed")

    # Test 2: Create sample RGB image
    sample_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    logger.info(f"Sample image shape: {sample_image.shape}")

    # Test 3: Test multi-channel image creation
    try:
        multi_channel = cnn_model._create_multi_channel_images(
            np.expand_dims(sample_image, axis=0))
        logger.info(f"Multi-channel image shape: {multi_channel.shape}")
        assert multi_channel.shape == (1, 128, 128, 6)
        logger.info("✓ Multi-channel image creation test passed")
    except Exception as e:
        logger.error(f"Multi-channel image creation failed: {e}")
        return False

    # Test 4: Test individual channel creation
    try:
        edge_map = cnn_model._create_edge_map(sample_image)
        enhanced_luminance = cnn_model._create_enhanced_luminance(sample_image)
        feature_map = cnn_model._create_feature_map(sample_image)

        assert edge_map.shape == (128, 128)
        assert enhanced_luminance.shape == (128, 128)
        assert feature_map.shape == (128, 128)
        logger.info("✓ Individual channel creation test passed")
    except Exception as e:
        logger.error(f"Individual channel creation failed: {e}")
        return False

    # Test 5: Test model building with multi-channel input
    try:
        cnn_model.num_classes = 5  # Set dummy number of classes
        model = cnn_model.build_model()
        logger.info(f"Model input shape: {model.input_shape}")
        assert model.input_shape == (None, 128, 128, 6)
        logger.info("✓ Model building test passed")
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        return False

    return True


def test_data_pipeline_multi_channel():
    """Test the data pipeline multi-channel functionality."""
    logger.info("Testing data pipeline multi-channel functionality...")

    # Initialize data pipeline
    data_pipeline = KolamDataPipeline(img_height=128, img_width=128)

    # Test 1: Check configuration
    logger.info(f"Data Pipeline Config: {data_pipeline.config}")

    # Test 2: Create sample images
    sample_images = np.random.randint(
        0, 255, (10, 128, 128, 3), dtype=np.uint8)
    sample_labels = np.random.randint(0, 3, 10)

    # Test 3: Test multi-channel preprocessing
    try:
        multi_channel_images = data_pipeline.preprocess_multi_channel(
            sample_images)
        logger.info(
            f"Multi-channel images shape: {multi_channel_images.shape}")
        assert multi_channel_images.shape == (10, 128, 128, 6)
        logger.info("✓ Data pipeline multi-channel preprocessing test passed")
    except Exception as e:
        logger.error(f"Data pipeline multi-channel preprocessing failed: {e}")
        return False

    return True


def test_shape_consistency():
    """Test that all components have consistent shapes."""
    logger.info("Testing shape consistency...")

    # Test with different image sizes
    test_sizes = [(64, 64), (128, 128), (224, 224)]

    for height, width in test_sizes:
        logger.info(f"Testing with image size: {height}x{width}")

        # Initialize model
        cnn_model = KolamCNNModel(img_height=height, img_width=width)
        cnn_model.num_classes = 5

        # Create sample image
        sample_image = np.random.randint(
            0, 255, (height, width, 3), dtype=np.uint8)

        # Test multi-channel creation
        multi_channel = cnn_model._create_multi_channel_images(
            np.expand_dims(sample_image, axis=0))
        expected_shape = (1, height, width, 6)

        assert multi_channel.shape == expected_shape, f"Expected {expected_shape}, got {multi_channel.shape}"

        # Test model building
        model = cnn_model.build_model()
        expected_input_shape = (None, height, width, 6)

        assert model.input_shape == expected_input_shape, f"Expected {expected_input_shape}, got {model.input_shape}"

        logger.info(f"✓ Shape consistency test passed for {height}x{width}")

    return True


def main():
    """Run all tests."""
    logger.info("Starting multi-channel CNN implementation tests...")

    tests = [
        ("Multi-channel preprocessing", test_multi_channel_preprocessing),
        ("Data pipeline multi-channel", test_data_pipeline_multi_channel),
        ("Shape consistency", test_shape_consistency)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")

        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")

    logger.info(f"\n{'='*50}")
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")

    if passed == total:
        logger.info(
            "🎉 All tests passed! Multi-channel implementation is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

