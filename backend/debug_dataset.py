"""
Debug Dataset Loading

This script helps debug why the dataset loading is not working.
"""

import os
import glob
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_dataset_loading():
    """
    Debug the dataset loading process.
    """
    print("Debugging dataset loading...")

    dataset_path = 'rangoli_dataset_complete/images'
    print(f"Dataset path: {dataset_path}")
    print(f"Path exists: {os.path.exists(dataset_path)}")

    if os.path.exists(dataset_path):
        # List all subdirectories
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(
            os.path.join(dataset_path, d))]
        print(f"Subdirectories: {subdirs}")

        # Count total files
        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, file))

        print(f"Total image files found: {len(all_files)}")

        if all_files:
            print("Sample files:")
            for f in all_files[:5]:
                print(f"  {f}")

                # Try to load the image
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        f, target_size=(64, 64))
                    print(f"    [OK] Loaded successfully: {img.size}")
                except Exception as e:
                    print(f"    [ERROR] Failed to load: {e}")

        # Test the class name generation
        print("\nTesting class name generation:")
        for root, dirs, files in os.walk(dataset_path):
            if files:  # Only process directories with files
                rel_path = os.path.relpath(root, dataset_path)
                if rel_path == '.':
                    class_name = 'root'
                else:
                    class_name = rel_path.replace('\\', '/').replace('/', '_')
                print(f"  {rel_path} -> {class_name}")

                # Count files in this directory
                image_files = [f for f in files if f.lower().endswith(
                    ('.jpg', '.jpeg', '.png'))]
                print(f"    Image files: {len(image_files)}")
                break  # Just test the first directory


if __name__ == "__main__":
    debug_dataset_loading()
