"""
Kolam Computer Vision Enhancement Module

This module provides advanced computer vision preprocessing techniques specifically
designed for kolam pattern recognition, including edge detection, noise cancellation,
and feature extraction to improve CNN performance.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, feature, morphology, measure
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, roberts
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KolamCVEnhancement:
    """
    Advanced computer vision preprocessing for kolam pattern recognition.
    """

    def __init__(self):
        """Initialize the computer vision enhancement module."""
        # Edge detection parameters
        self.edge_detection_params = {
            'canny_low': 50,
            'canny_high': 150,
            'sobel_kernel': 3,
            'prewitt_kernel': 3
        }

        # Noise reduction parameters
        self.noise_reduction_params = {
            'bilateral_d': 9,
            'bilateral_sigma': 75,
            'median_kernel': 3,
            'gaussian_kernel': (5, 5),
            'morph_kernel': (3, 3)
        }

        # Pattern enhancement parameters
        self.pattern_params = {
            'adaptive_block': 11,
            'adaptive_c': 2,
            'clahe_clip': 2.0,
            'clahe_grid': (8, 8)
        }

    def preprocess_image(self, image_path: str) -> Dict:
        """
        Complete preprocessing pipeline for a single image.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing all preprocessing results
        """
        logger.info(f"Preprocessing image: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Noise removal
        denoised = self.remove_noise(image)

        # Step 2: Edge detection
        edges = self.detect_edges(denoised)

        # Step 3: Pattern enhancement
        enhanced = self.enhance_patterns(denoised)

        # Step 4: Feature extraction
        features = self.extract_features(enhanced)

        # Step 5: Multi-channel combination
        combined = self.combine_features(image, edges, enhanced, features)

        results = {
            'original': image,
            'denoised': denoised,
            'edges': edges,
            'enhanced': enhanced,
            'features': features,
            'combined': combined,
            'metadata': {
                'original_shape': image.shape,
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'algorithm_version': '1.0'
            }
        }

        logger.info("Image preprocessing completed successfully")
        return results

    def preprocess_image_from_array(self, image: np.ndarray) -> Dict:
        """
        Preprocess image from numpy array instead of file path.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            Dictionary containing preprocessing results
        """
        # Step 1: Noise removal
        denoised = self.remove_noise(image)

        # Step 2: Edge detection
        edges = self.detect_edges(denoised)

        # Step 3: Pattern enhancement
        enhanced = self.enhance_patterns(denoised)

        # Step 4: Feature extraction
        features = self.extract_features(enhanced)

        # Step 5: Multi-channel combination
        combined = self.combine_features(image, edges, enhanced, features)

        results = {
            'original': image,
            'denoised': denoised,
            'edges': edges,
            'enhanced': enhanced,
            'features': features,
            'combined': combined,
            'metadata': {
                'original_shape': image.shape,
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'algorithm_version': '1.0'
            }
        }

        return results

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-stage noise removal optimized for kolam patterns.

        Args:
            image: Input RGB image

        Returns:
            Denoised image
        """
        # Convert to float for processing
        image_float = image.astype(np.float32) / 255.0

        # Step 1: Bilateral filtering for edge-preserving smoothing
        bilateral = cv2.bilateralFilter(
            (image_float * 255).astype(np.uint8),
            self.noise_reduction_params['bilateral_d'],
            self.noise_reduction_params['bilateral_sigma'],
            self.noise_reduction_params['bilateral_sigma']
        )

        # Step 2: Median blur for salt-and-pepper noise
        median = cv2.medianBlur(
            bilateral, self.noise_reduction_params['median_kernel'])

        # Step 3: Morphological operations for texture noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.noise_reduction_params['morph_kernel']
        )

        # Opening to remove small objects
        opened = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)

        # Closing to fill small holes
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced edge detection combining multiple algorithms.

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            Edge-detected image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Method 1: Canny edge detection
        canny_edges = cv2.Canny(
            gray,
            self.edge_detection_params['canny_low'],
            self.edge_detection_params['canny_high']
        )

        # Method 2: Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)

        # Guard against division by zero
        max_val = sobel_edges.max()
        if max_val > 0:
            sobel_edges = (sobel_edges / max_val * 255).astype(np.uint8)
        else:
            sobel_edges = np.zeros_like(sobel_edges, dtype=np.uint8)

        # Method 3: Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Combine edges using weighted average
        combined_edges = (
            canny_edges * 0.5 +
            (sobel_edges > 100).astype(np.uint8) * 255 * 0.3 +
            gradient * 0.2
        ).astype(np.uint8)

        # Final cleanup
        final_edges = cv2.medianBlur(combined_edges, 3)

        return final_edges

    def enhance_patterns(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance kolam patterns using adaptive thresholding and CLAHE.

        Args:
            image: Input image

        Returns:
            Pattern-enhanced image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Method 1: Adaptive thresholding for varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.pattern_params['adaptive_block'],
            self.pattern_params['adaptive_c']
        )

        # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.pattern_params['clahe_clip'],
            tileGridSize=self.pattern_params['clahe_grid']
        )
        clahe_enhanced = clahe.apply(gray)

        # Method 3: Morphological enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_enhanced = cv2.morphologyEx(
            clahe_enhanced, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # Combine enhancements
        enhanced = np.maximum(adaptive_thresh, morph_enhanced)

        return enhanced

    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract structural features from kolam patterns.

        Args:
            image: Input binary/enhanced image

        Returns:
            Dictionary of extracted features
        """
        # Ensure binary image
        if image.max() > 1:
            binary = (image > 127).astype(np.uint8) * 255
        else:
            binary = (image > 0.5).astype(np.uint8) * 255

        # Feature extraction using skimage
        # Label connected components
        labeled_image, num_features = measure.label(binary, return_num=True)

        # Region properties
        regions = measure.regionprops(labeled_image)

        # Calculate features
        features = {
            'num_components': num_features,
            'total_area': np.sum(binary > 0),
            'component_sizes': [region.area for region in regions],
            'component_perimeters': [region.perimeter for region in regions],
            'component_eccentricities': [region.eccentricity for region in regions],
            'component_solidities': [region.solidity for region in regions]
        }

        # Detect lines and circles
        lines = self._detect_lines(binary)
        circles = self._detect_circles(binary)

        features.update({
            'detected_lines': lines,
            'detected_circles': circles,
            'symmetry_score': self._calculate_symmetry(binary)
        })

        return features

    def _detect_lines(self, binary: np.ndarray) -> list:
        """Detect lines in the binary image using HoughLinesP."""
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180,
                                threshold=80, minLineLength=30, maxLineGap=10)
        if lines is not None:
            return [tuple(line[0]) for line in lines]
        return []

    def _detect_circles(self, binary: np.ndarray) -> list:
        """Detect circles in the binary image using HoughCircles."""
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(x, y, r) for (x, y, r) in circles]
        return []

    def _calculate_symmetry(self, binary_image: np.ndarray) -> float:
        """Calculate symmetry score of the pattern."""
        # Flip horizontally and vertically
        h_flip = np.fliplr(binary_image)
        v_flip = np.flipud(binary_image)

        # Calculate similarity scores
        h_similarity = np.mean(np.abs(binary_image.astype(
            np.float32) - h_flip.astype(np.float32))) / 255.0
        v_similarity = np.mean(np.abs(binary_image.astype(
            np.float32) - v_flip.astype(np.float32))) / 255.0

        # Combine similarities (lower values = more symmetric)
        symmetry_score = 1.0 - (h_similarity + v_similarity) / 2.0

        return max(0.0, symmetry_score)

    def combine_features(self, original: np.ndarray, edges: np.ndarray,
                         enhanced: np.ndarray, features: Dict) -> np.ndarray:
        """
        Combine multiple feature channels into multi-channel image.

        Args:
            original: Original RGB image
            edges: Edge-detected image
            enhanced: Pattern-enhanced image
            features: Extracted features dictionary

        Returns:
            Multi-channel image combining all features
        """
        # Normalize all channels to [0, 255]
        original_norm = original.astype(np.float32)

        # Edge channel (single channel)
        edges_norm = edges.astype(np.float32)
        if edges_norm.max() > 0:
            edges_norm = (edges_norm / edges_norm.max()) * 255

        # Enhanced pattern channel (single channel)
        enhanced_norm = enhanced.astype(np.float32)
        if enhanced_norm.max() > 0:
            enhanced_norm = (enhanced_norm / enhanced_norm.max()) * 255

        # Create feature map based on extracted features
        feature_map = self._create_feature_map(original.shape[:2], features)

        # Combine into multi-channel image
        # Channel 1-3: Original RGB
        # Channel 4: Edge features
        # Channel 5: Enhanced patterns
        # Channel 6: Feature map
        combined = np.zeros(
            (original.shape[0], original.shape[1], 6), dtype=np.float32)

        combined[:, :, :3] = original_norm / 255.0  # Normalize to [0, 1]
        combined[:, :, 3] = edges_norm / 255.0
        combined[:, :, 4] = enhanced_norm / 255.0
        combined[:, :, 5] = feature_map / 255.0

        return combined

    def _create_feature_map(self, shape: Tuple, features: Dict) -> np.ndarray:
        """Create a feature map visualization."""
        feature_map = np.zeros(shape, dtype=np.float32)

        # Add component density
        if features['num_components'] > 0:
            # Create density map based on component sizes
            total_area = features['total_area']
            if total_area > 0:
                density = features['num_components'] / total_area
                feature_map.fill(density * 1000)  # Scale for visibility

        # Add symmetry information
        symmetry = features.get('symmetry_score', 0)
        feature_map += symmetry * 100

        return np.clip(feature_map, 0, 255)

    def batch_preprocess(self, image_paths: List[str]) -> List[Dict]:
        """
        Preprocess multiple images in batch.

        Args:
            image_paths: List of image file paths

        Returns:
            List of preprocessing results
        """
        logger.info(f"Batch preprocessing {len(image_paths)} images")

        results = []
        for i, path in enumerate(image_paths):
            try:
                result = self.preprocess_image(path)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")

            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                continue

        logger.info(
            f"Batch preprocessing completed: {len(results)} successful")
        return results

    def visualize_preprocessing(self, results: Dict, save_path: str = None):
        """
        Create visualization of preprocessing steps.

        Args:
            results: Preprocessing results dictionary
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Kolam Pattern Preprocessing Pipeline', fontsize=16)

        # Original image
        axes[0, 0].imshow(results['original'])
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # Denoised image
        axes[0, 1].imshow(results['denoised'])
        axes[0, 1].set_title('Noise Removed')
        axes[0, 1].axis('off')

        # Edge detection
        axes[0, 2].imshow(results['edges'], cmap='gray')
        axes[0, 2].set_title('Edge Detection')
        axes[0, 2].axis('off')

        # Pattern enhancement
        axes[1, 0].imshow(results['enhanced'], cmap='gray')
        axes[1, 0].set_title('Pattern Enhanced')
        axes[1, 0].axis('off')

        # Feature map
        if 'features' in results:
            feature_map = self._create_feature_map(
                results['original'].shape[:2],
                results['features']
            )
            axes[1, 1].imshow(feature_map, cmap='hot')
            axes[1, 1].set_title('Feature Map')
            axes[1, 1].axis('off')

        # Combined multi-channel
        axes[1, 2].imshow(results['combined'][:, :, :3])
        axes[1, 2].set_title('Multi-channel Output')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Preprocessing visualization saved to {save_path}")

        return fig


def main():
    """Demonstrate the computer vision enhancement module."""
    # Initialize the enhancer
    enhancer = KolamCVEnhancement()

    # Test with a sample image (if available)
    test_image = 'static/mandalaKolam.jpg'  # Using existing kolam image

    if os.path.exists(test_image):
        print("Testing computer vision enhancement...")

        # Process the image
        results = enhancer.preprocess_image(test_image)

        # Create visualization
        enhancer.visualize_preprocessing(
            results, 'backend/cv_enhancement_demo.png')

        print("Computer vision enhancement completed successfully!")
        print(f"Processed image shape: {results['original'].shape}")
        print(
            f"Number of components detected: {results['features']['num_components']}")
        print(f"Symmetry score: {results['features']['symmetry_score']:.3f}")
    else:
        print(f"Test image not found: {test_image}")
        print("Computer vision enhancement module is ready for use!")


if __name__ == "__main__":
    main()
