"""
Kolam Advanced Symmetry Analysis and Mathematical Property Extraction

This module performs comprehensive mathematical analysis of kolam patterns including
symmetry detection, fractal dimension calculation, tessellation analysis, and
advanced geometric property extraction for intelligent pattern regeneration.
"""

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage import measure, transform, filters, feature, morphology
from skimage.transform import rotate as skimage_rotate
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymmetryType(Enum):
    """Types of symmetry in kolam patterns."""
    REFLECTION_HORIZONTAL = "reflection_horizontal"
    REFLECTION_VERTICAL = "reflection_vertical"
    REFLECTION_DIAGONAL = "reflection_diagonal"
    ROTATIONAL_2_FOLD = "rotational_2_fold"
    ROTATIONAL_4_FOLD = "rotational_4_fold"
    ROTATIONAL_8_FOLD = "rotational_8_fold"
    RADIAL = "radial"
    TRANSLATIONAL = "translational"
    FRACTAL_SELF_SIMILARITY = "fractal_self_similarity"


@dataclass
class SymmetryAnalysis:
    """Results of symmetry analysis."""
    symmetry_type: SymmetryType
    confidence_score: float
    symmetry_order: int
    symmetry_angle: float
    symmetry_center: Tuple[float, float]
    symmetry_strength: float
    violation_points: List[Tuple[int, int]]


@dataclass
class MathematicalProperties:
    """Mathematical properties of kolam pattern."""
    fractal_dimension: float
    lacunarity: float
    correlation_dimension: float
    lyapunov_exponent: float
    tessellation_order: int
    grid_complexity: float
    motif_count: int
    connectivity_index: float
    curvature_measure: float
    aspect_ratio: float


class KolamSymmetryAnalyzer:
    """
    Advanced symmetry analysis and mathematical property extraction for kolam patterns.
    """

    def __init__(self):
        """Initialize the symmetry analyzer."""
        self.symmetry_tolerance = 0.05  # 5% tolerance for symmetry detection
        self.min_symmetry_strength = 0.7  # Minimum strength for valid symmetry
        self.fractal_box_sizes = [2, 4, 8, 16, 32, 64, 128]

    def _ensure_binary(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image is binary (0/255 uint8 format).

        Args:
            image: Input image (grayscale or binary)

        Returns:
            Binary image as uint8 with values 0 or 255
        """
        # Handle different input types
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Convert to binary
        if image.dtype == np.bool_:
            return image.astype(np.uint8) * 255
        elif image.max() > 1:
            return (image > 127).astype(np.uint8) * 255
        else:
            return (image > 0.5).astype(np.uint8) * 255

    def analyze_pattern_symmetries(self, image: np.ndarray) -> Dict[SymmetryType, SymmetryAnalysis]:
        """
        Perform comprehensive symmetry analysis on kolam pattern.

        Args:
            image: Binary or grayscale image of kolam pattern

        Returns:
            Dictionary of symmetry analyses by type
        """
        logger.info("Performing comprehensive symmetry analysis...")

        # Ensure binary image
        binary = self._ensure_binary(image)

        symmetries_found = {}

        # Test reflection symmetries
        reflection_symmetries = self._analyze_reflection_symmetries(binary)
        symmetries_found.update(reflection_symmetries)

        # Test rotational symmetries
        rotational_symmetries = self._analyze_rotational_symmetries(binary)
        symmetries_found.update(rotational_symmetries)

        # Test radial symmetry
        radial_symmetry = self._analyze_radial_symmetry(binary)
        if radial_symmetry:
            symmetries_found[SymmetryType.RADIAL] = radial_symmetry

        # Test translational symmetry
        translational_symmetry = self._analyze_translational_symmetry(binary)
        if translational_symmetry:
            symmetries_found[SymmetryType.TRANSLATIONAL] = translational_symmetry

        logger.info(
            f"Symmetry analysis completed. Found {len(symmetries_found)} symmetry types")
        return symmetries_found

    def _analyze_reflection_symmetries(self, binary: np.ndarray) -> Dict[SymmetryType, SymmetryAnalysis]:
        """Analyze reflection symmetries (horizontal, vertical, diagonal)."""
        symmetries = {}
        height, width = binary.shape

        # Horizontal reflection
        if height > 10 and width > 10:
            h_analysis = self._test_reflection_symmetry(binary, 'horizontal')
            if h_analysis.confidence_score >= self.min_symmetry_strength:
                symmetries[SymmetryType.REFLECTION_HORIZONTAL] = h_analysis

        # Vertical reflection
        if height > 10 and width > 10:
            v_analysis = self._test_reflection_symmetry(binary, 'vertical')
            if v_analysis.confidence_score >= self.min_symmetry_strength:
                symmetries[SymmetryType.REFLECTION_VERTICAL] = v_analysis

        # Diagonal reflection (if square-ish)
        if abs(height - width) < min(height, width) * 0.2:
            d_analysis = self._test_reflection_symmetry(binary, 'diagonal')
            if d_analysis.confidence_score >= self.min_symmetry_strength:
                symmetries[SymmetryType.REFLECTION_DIAGONAL] = d_analysis

        return symmetries

    def _test_reflection_symmetry(self, binary: np.ndarray, axis: str) -> SymmetryAnalysis:
        """Test reflection symmetry along specified axis."""
        height, width = binary.shape

        if axis == 'horizontal':
            # Flip top-bottom
            flipped = np.flipud(binary)
            center = (height // 2, width // 2)
        elif axis == 'vertical':
            # Flip left-right
            flipped = np.fliplr(binary)
            center = (height // 2, width // 2)
        elif axis == 'diagonal':
            # Transpose for diagonal reflection
            if height != width:
                # Resize to square for diagonal comparison
                size = min(height, width)
                binary_square = cv2.resize(binary, (size, size))
                flipped = binary_square.T
            else:
                flipped = binary.T
            center = (height // 2, width // 2)
        else:
            raise ValueError(f"Unknown axis: {axis}")

        # Ensure same shape for comparison
        if binary.shape != flipped.shape:
            flipped = cv2.resize(flipped, (width, height))

        # Calculate similarity
        difference = np.abs(binary.astype(np.float32) -
                            flipped.astype(np.float32))
        total_pixels = binary.shape[0] * binary.shape[1]
        if total_pixels > 0:
            similarity = 1.0 - (np.sum(difference) / (255.0 * total_pixels))
        else:
            similarity = 0.0

        # Find violation points (areas with low similarity)
        violation_mask = difference > (255 * (1 - self.symmetry_tolerance))
        violation_points = np.where(violation_mask)

        return SymmetryAnalysis(
            symmetry_type=SymmetryType[f"REFLECTION_{axis.upper()}"],
            confidence_score=similarity,
            symmetry_order=2,
            symmetry_angle=0.0,
            symmetry_center=center,
            symmetry_strength=similarity,
            violation_points=list(
                # Limit violations
                zip(violation_points[0][:100], violation_points[1][:100]))
        )

    def _analyze_rotational_symmetries(self, binary: np.ndarray) -> Dict[SymmetryType, SymmetryAnalysis]:
        """Analyze rotational symmetries (2-fold, 4-fold, 8-fold)."""
        symmetries = {}

        # Test different rotation orders
        for order in [2, 4, 8]:
            angle = 360.0 / order
            r_analysis = self._test_rotational_symmetry(binary, order, angle)
            if r_analysis.confidence_score >= self.min_symmetry_strength:
                symmetry_type = SymmetryType[f"ROTATIONAL_{order}_FOLD"]
                symmetries[symmetry_type] = r_analysis

        return symmetries

    def _test_rotational_symmetry(self, binary: np.ndarray, order: int, angle: float) -> SymmetryAnalysis:
        """Test rotational symmetry of given order."""
        height, width = binary.shape
        center_y, center_x = height // 2, width // 2

        # Create rotated versions
        rotated_images = []
        for i in range(1, order):
            angle_i = angle * i
            # Use cv2 for rotation which is more reliable
            rotation_matrix = cv2.getRotationMatrix2D(
                (center_x, center_y), angle_i, 1.0)
            rotated = cv2.warpAffine(binary, rotation_matrix, (width, height))
            rotated = (rotated > 127).astype(np.uint8) * 255
            rotated_images.append(rotated)

        # Compare all rotations with original
        similarities = []
        for rotated in rotated_images:
            difference = np.abs(binary.astype(
                np.float32) - rotated.astype(np.float32))
            total_pixels = binary.shape[0] * binary.shape[1]
            if total_pixels > 0:
                similarity = 1.0 - (np.sum(difference) /
                                    (255.0 * total_pixels))
            else:
                similarity = 0.0
            similarities.append(similarity)

        # Average similarity across all rotations
        avg_similarity = np.mean(similarities) if similarities else 0.0

        # Find consistent violation points across rotations
        violation_points = []
        sample_step = 5  # Sample every 5th pixel for efficiency
        for y in range(0, height, sample_step):
            for x in range(0, width, sample_step):
                pixel_values = [binary[y, x]]
                for rotated in rotated_images:
                    pixel_values.append(rotated[y, x])

                # Check if pixel is inconsistent across rotations
                if np.std(pixel_values) > (255 * (1 - self.symmetry_tolerance)):
                    violation_points.append((y, x))

        return SymmetryAnalysis(
            symmetry_type=SymmetryType[f"ROTATIONAL_{order}_FOLD"],
            confidence_score=avg_similarity,
            symmetry_order=order,
            symmetry_angle=angle,
            symmetry_center=(center_y, center_x),
            symmetry_strength=avg_similarity,
            violation_points=violation_points[:100]  # Limit violations
        )

    def _analyze_radial_symmetry(self, binary: np.ndarray) -> Optional[SymmetryAnalysis]:
        """Analyze radial symmetry around center point."""
        height, width = binary.shape
        center_y, center_x = height // 2, width // 2

        # Test multiple angles for radial symmetry
        test_angles = [30, 45, 60, 90, 120, 180]
        similarities = []

        for angle in test_angles:
            # Create radially symmetric test pattern
            radial_test = self._create_radial_test_pattern(
                binary, center_y, center_x, angle)

            # Compare with original
            difference = np.abs(binary.astype(np.float32) -
                                radial_test.astype(np.float32))
            total_pixels = binary.shape[0] * binary.shape[1]
            if total_pixels > 0:
                similarity = 1.0 - (np.sum(difference) /
                                    (255.0 * total_pixels))
            else:
                similarity = 0.0
            similarities.append(similarity)

        avg_similarity = np.mean(similarities) if similarities else 0.0

        if avg_similarity >= self.min_symmetry_strength:
            return SymmetryAnalysis(
                symmetry_type=SymmetryType.RADIAL,
                confidence_score=avg_similarity,
                symmetry_order=len(test_angles),
                symmetry_angle=360.0 / len(test_angles),
                symmetry_center=(center_y, center_x),
                symmetry_strength=avg_similarity,
                violation_points=[]
            )

        return None

    def _create_radial_test_pattern(self, binary: np.ndarray, center_y: int, center_x: int, angle: float) -> np.ndarray:
        """Create a test pattern for radial symmetry analysis."""
        height, width = binary.shape
        test_pattern = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Calculate angle from center
                dy = y - center_y
                dx = x - center_x
                if dx == 0 and dy == 0:
                    test_pattern[y, x] = binary[y, x]
                    continue

                original_angle = math.atan2(dy, dx) * 180 / math.pi
                # Test if pattern repeats every 'angle' degrees
                test_angle = (original_angle // angle) * angle
                test_radians = test_angle * math.pi / 180

                # Find corresponding point
                distance = math.sqrt(dx*dx + dy*dy)
                test_x = center_x + int(distance * math.cos(test_radians))
                test_y = center_y + int(distance * math.sin(test_radians))

                # Copy value if within bounds
                if 0 <= test_x < width and 0 <= test_y < height:
                    test_pattern[y, x] = binary[test_y, test_x]

        return test_pattern

    def _analyze_translational_symmetry(self, binary: np.ndarray) -> Optional[SymmetryAnalysis]:
        """Analyze translational symmetry (repeating patterns)."""
        try:
            # Use autocorrelation to detect periodic patterns
            autocorr = self._calculate_autocorrelation(binary)

            # Find peaks in autocorrelation (indicating repetition)
            peaks = self._find_autocorrelation_peaks(autocorr)

            if len(peaks) >= 2:
                # Calculate average translation vector
                vectors = []
                for i in range(1, min(len(peaks), 5)):  # Limit to first 5 peaks
                    prev_peak = peaks[i-1]
                    curr_peak = peaks[i]
                    vector = (curr_peak[0] - prev_peak[0],
                              curr_peak[1] - prev_peak[1])
                    vectors.append(vector)

                if vectors:
                    avg_vector = np.mean(vectors, axis=0)

                    # Calculate confidence from peak strengths
                    autocorr_range = autocorr.max() - autocorr.min()
                    if autocorr_range > 0:
                        autocorr_norm = (
                            autocorr - autocorr.min()) / autocorr_range
                    else:
                        autocorr_norm = np.zeros_like(autocorr)

                    # Calculate peak strengths from normalized autocorrelation
                    peak_strengths = []
                    for p in peaks[:5]:  # Limit to first 5 peaks
                        if 0 <= p[0] < autocorr_norm.shape[0] and 0 <= p[1] < autocorr_norm.shape[1]:
                            peak_strengths.append(autocorr_norm[p[0], p[1]])

                    confidence = np.mean(
                        peak_strengths) if peak_strengths else 0.0

                    return SymmetryAnalysis(
                        symmetry_type=SymmetryType.TRANSLATIONAL,
                        confidence_score=confidence,
                        symmetry_order=len(peaks),
                        symmetry_angle=math.atan2(
                            avg_vector[1], avg_vector[0]) * 180 / math.pi,
                        symmetry_center=tuple(peaks[0]),
                        symmetry_strength=confidence,
                        violation_points=[]
                    )
        except Exception as e:
            logger.warning(f"Error in translational symmetry analysis: {e}")

        return None

    def _calculate_autocorrelation(self, binary: np.ndarray) -> np.ndarray:
        """Calculate 2D autocorrelation of binary image."""
        try:
            # Normalize image
            binary_float = binary.astype(np.float32) / 255.0

            # Calculate autocorrelation using FFT
            fft = np.fft.fft2(binary_float)
            autocorr_fft = fft * np.conj(fft)
            autocorr = np.fft.ifft2(autocorr_fft).real

            # Shift zero frequency to center
            autocorr = np.fft.fftshift(autocorr)

            return autocorr
        except Exception as e:
            logger.warning(f"Error calculating autocorrelation: {e}")
            return np.zeros_like(binary, dtype=np.float32)

    def _find_autocorrelation_peaks(self, autocorr: np.ndarray) -> List[Tuple[int, int]]:
        """Find significant peaks in autocorrelation."""
        try:
            # Normalize autocorrelation
            if autocorr.max() > autocorr.min():
                autocorr_norm = (autocorr - autocorr.min()) / \
                    (autocorr.max() - autocorr.min())
            else:
                return []

            # Find local maxima above threshold
            peaks = []
            threshold = 0.7

            # Use scipy for peak detection
            from scipy import ndimage

            # Apply Gaussian filter to smooth
            autocorr_smooth = ndimage.gaussian_filter(autocorr_norm, sigma=1)

            # Find local maxima
            local_maxima = (autocorr_smooth == ndimage.maximum_filter(
                autocorr_smooth, size=5))

            # Get coordinates of maxima above threshold
            coords = np.where(local_maxima & (autocorr_smooth > threshold))

            for y, x in zip(coords[0], coords[1]):
                peaks.append((y, x))

            # Sort by strength
            peaks.sort(key=lambda p: autocorr_norm[p[0], p[1]], reverse=True)

            return peaks[:10]  # Return top 10 peaks

        except Exception as e:
            logger.warning(f"Error finding autocorrelation peaks: {e}")
            return []

    def calculate_mathematical_properties(self, image: np.ndarray) -> MathematicalProperties:
        """
        Calculate comprehensive mathematical properties of kolam pattern.

        Args:
            image: Binary or grayscale image of kolam pattern

        Returns:
            Mathematical properties of the pattern
        """
        logger.info("Calculating mathematical properties...")

        # Ensure binary image for analysis
        binary = self._ensure_binary(image)

        # Calculate fractal dimension using box counting
        fractal_dim = self._calculate_fractal_dimension(binary)

        # Calculate lacunarity (texture/gap measure)
        lacunarity = self._calculate_lacunarity(binary)

        # Calculate correlation dimension
        correlation_dim = self._calculate_correlation_dimension(binary)

        # Calculate Lyapunov exponent (chaos measure)
        lyapunov_exp = self._calculate_lyapunov_exponent(binary)

        # Analyze tessellation properties
        tessellation_order = self._analyze_tessellation(binary)

        # Calculate grid complexity
        grid_complexity = self._calculate_grid_complexity(binary)

        # Count motifs and patterns
        motif_count = self._count_motifs(binary)

        # Calculate connectivity index
        connectivity_index = self._calculate_connectivity_index(binary)

        # Calculate curvature measure
        curvature_measure = self._calculate_curvature_measure(binary)

        # Calculate aspect ratio
        aspect_ratio = self._calculate_aspect_ratio(binary)

        return MathematicalProperties(
            fractal_dimension=fractal_dim,
            lacunarity=lacunarity,
            correlation_dimension=correlation_dim,
            lyapunov_exponent=lyapunov_exp,
            tessellation_order=tessellation_order,
            grid_complexity=grid_complexity,
            motif_count=motif_count,
            connectivity_index=connectivity_index,
            curvature_measure=curvature_measure,
            aspect_ratio=aspect_ratio
        )

    def _calculate_fractal_dimension(self, binary: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method."""
        try:
            # Box counting algorithm
            sizes = []
            counts = []

            for box_size in self.fractal_box_sizes:
                if box_size > min(binary.shape):
                    continue

                # Count non-empty boxes
                count = 0
                for y in range(0, binary.shape[0], box_size):
                    for x in range(0, binary.shape[1], box_size):
                        box = binary[y:y+box_size, x:x+box_size]
                        if np.any(box > 0):
                            count += 1

                if count > 0:
                    sizes.append(box_size)
                    counts.append(count)

            if len(sizes) >= 2:
                # Linear regression in log-log space
                log_sizes = np.log(sizes)
                log_counts = np.log(counts)

                # Simple linear fit
                coeffs = np.polyfit(log_sizes, log_counts, 1)
                fractal_dimension = -coeffs[0]

                # Clamp to reasonable range [1.0, 2.0]
                return np.clip(fractal_dimension, 1.0, 2.0)

        except Exception as e:
            logger.warning(f"Error calculating fractal dimension: {e}")

        return 1.5  # Default value

    def _calculate_lacunarity(self, binary: np.ndarray) -> float:
        """Calculate lacunarity (measure of texture/gappiness)."""
        try:
            # Use gliding box algorithm
            box_sizes = [5, 10, 15, 20]
            lacunarities = []

            for box_size in box_sizes:
                if box_size >= min(binary.shape):
                    continue

                masses = []
                for y in range(binary.shape[0] - box_size + 1):
                    for x in range(binary.shape[1] - box_size + 1):
                        box = binary[y:y+box_size, x:x+box_size]
                        mass = np.sum(box > 0)
                        masses.append(mass)

                if masses:
                    masses = np.array(masses, dtype=np.float32)
                    mean_mass = np.mean(masses)
                    var_mass = np.var(masses)

                    if mean_mass > 0:
                        lacunarity = (var_mass / (mean_mass ** 2)) + 1
                        lacunarities.append(lacunarity)

            return np.mean(lacunarities) if lacunarities else 1.0

        except Exception as e:
            logger.warning(f"Error calculating lacunarity: {e}")
            return 1.0

    def _calculate_correlation_dimension(self, binary: np.ndarray) -> float:
        """Calculate correlation dimension using correlation integral."""
        try:
            # Get pattern points
            points = np.where(binary > 0)
            if len(points[0]) < 10:
                return 1.0

            # Sample points for efficiency
            max_points = 500
            if len(points[0]) > max_points:
                indices = np.random.choice(
                    len(points[0]), max_points, replace=False)
                y_coords = points[0][indices]
                x_coords = points[1][indices]
            else:
                y_coords = points[0]
                x_coords = points[1]

            # Calculate pairwise distances efficiently
            coords = np.column_stack([y_coords, x_coords])
            from scipy.spatial.distance import pdist
            distances = pdist(coords)

            if len(distances) == 0:
                return 1.0

            # Correlation integral for different radii
            radii = np.logspace(0, 2, 10)
            correlation_values = []

            for r in radii:
                # Count pairs within radius r
                within_r = np.sum(distances <= r)
                total_pairs = len(distances)
                if total_pairs > 0:
                    correlation = within_r / total_pairs
                    if correlation > 0:
                        correlation_values.append(correlation)

            if len(correlation_values) >= 2:
                # Fit power law: C(r) ~ r^D
                valid_radii = radii[:len(correlation_values)]
                log_r = np.log(valid_radii)
                log_c = np.log(np.array(correlation_values))

                # Remove inf and nan values
                mask = np.isfinite(log_r) & np.isfinite(log_c)
                if np.sum(mask) >= 2:
                    coeffs = np.polyfit(log_r[mask], log_c[mask], 1)
                    return np.clip(coeffs[0], 1.0, 2.0)

        except Exception as e:
            logger.warning(f"Error calculating correlation dimension: {e}")

        return 1.5  # Default value

    def _calculate_lyapunov_exponent(self, binary: np.ndarray) -> float:
        """Calculate Lyapunov exponent (measure of chaos/sensitivity)."""
        try:
            # Simplified Lyapunov exponent calculation
            # Create slightly perturbed version
            noise = np.random.random(binary.shape) * 0.1
            perturbed = np.clip(binary.astype(
                np.float32) / 255.0 + noise, 0, 1)

            # Calculate divergence
            original = binary.astype(np.float32) / 255.0
            difference = np.abs(original - perturbed)
            avg_difference = np.mean(difference)

            # Lyapunov exponent approximation
            if avg_difference > 0 and avg_difference < 1:
                lyapunov_exp = -math.log(1 - avg_difference)
            else:
                lyapunov_exp = 0.1

            return np.clip(lyapunov_exp, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating Lyapunov exponent: {e}")
            return 0.1

    def _analyze_tessellation(self, binary: np.ndarray) -> int:
        """Analyze tessellation properties."""
        try:
            # Count connected components
            labeled, num_components = measure.label(
                binary > 0, return_num=True, connectivity=2)

            # Estimate tessellation order based on component distribution
            if num_components <= 4:
                return 1
            elif num_components <= 16:
                return 2
            elif num_components <= 64:
                return 3
            else:
                return 4

        except Exception as e:
            logger.warning(f"Error analyzing tessellation: {e}")
            return 1

    def _calculate_grid_complexity(self, binary: np.ndarray) -> float:
        """Calculate grid complexity measure."""
        try:
            # Analyze grid structure
            edges = cv2.Canny(binary, 50, 150)
            edge_density = np.sum(edges > 0) / \
                (edges.shape[0] * edges.shape[1])

            # Calculate line complexity
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 30, minLineLength=10, maxLineGap=5)

            if lines is not None:
                line_count = len(lines)
                # Calculate average line length
                line_lengths = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    line_lengths.append(length)
                avg_line_length = np.mean(line_lengths) if line_lengths else 0
            else:
                line_count = 0
                avg_line_length = 0

            # Normalize and combine metrics
            max_dimension = max(binary.shape)
            normalized_length = avg_line_length / max_dimension if max_dimension > 0 else 0

            complexity = (
                edge_density * 0.5 +
                min(line_count / 100, 1.0) * 0.3 +
                normalized_length * 0.2
            )

            return np.clip(complexity, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating grid complexity: {e}")
            return 0.5

    def _count_motifs(self, binary: np.ndarray) -> int:
        """Count distinct motifs in the pattern."""
        try:
            # Label connected components
            labeled, num_components = measure.label(
                binary > 0, return_num=True, connectivity=2)

            # Filter out very small components (noise)
            min_size = 10
            significant_components = 0

            for i in range(1, num_components + 1):
                component_mask = (labeled == i)
                if np.sum(component_mask) >= min_size:
                    significant_components += 1

            return significant_components

        except Exception as e:
            logger.warning(f"Error counting motifs: {e}")
            return 1

    def _calculate_connectivity_index(self, binary: np.ndarray) -> float:
        """Calculate connectivity index (how connected the pattern is)."""
        try:
            # Use morphological operations to analyze connectivity
            kernel = np.ones((3, 3), np.uint8)

            # Erosion to find core structure
            eroded = cv2.erode(binary, kernel, iterations=1)

            # Calculate connectivity as ratio of eroded to original
            original_mass = np.sum(binary > 0)
            eroded_mass = np.sum(eroded > 0)

            if original_mass > 0:
                connectivity = eroded_mass / original_mass
                return connectivity

            return 0.0

        except Exception as e:
            logger.warning(f"Error calculating connectivity index: {e}")
            return 0.5

    def _calculate_curvature_measure(self, binary: np.ndarray) -> float:
        """Calculate overall curvature measure of the pattern."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            # Calculate curvature for each contour
            total_curvature = 0.0
            total_length = 0.0

            for contour in contours:
                if len(contour) < 3:
                    continue

                # Calculate curvature using angle changes
                curvature_sum = 0.0
                for i in range(len(contour)):
                    p1 = contour[i][0]
                    p2 = contour[(i + 1) % len(contour)][0]
                    p3 = contour[(i + 2) % len(contour)][0]

                    # Calculate vectors
                    v1 = p1 - p2
                    v2 = p3 - p2

                    # Calculate angle
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)

                    if norm1 > 0 and norm2 > 0:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = math.acos(cos_angle)
                        curvature_sum += abs(angle)

                total_curvature += curvature_sum
                total_length += len(contour)

            if total_length > 0:
                avg_curvature = total_curvature / total_length
                return np.clip(avg_curvature / math.pi, 0.0, 1.0)

            return 0.0

        except Exception as e:
            logger.warning(f"Error calculating curvature measure: {e}")
            return 0.5

    def _calculate_aspect_ratio(self, binary: np.ndarray) -> float:
        """Calculate aspect ratio of the pattern."""
        try:
            # Find bounding box
            points = np.where(binary > 0)
            if len(points[0]) == 0:
                return 1.0

            min_y, max_y = np.min(points[0]), np.max(points[0])
            min_x, max_x = np.min(points[1]), np.max(points[1])

            height = max_y - min_y + 1
            width = max_x - min_x + 1

            if height > 0:
                return width / height

            return 1.0

        except Exception as e:
            logger.warning(f"Error calculating aspect ratio: {e}")
            return 1.0

    def _calculate_line_intersection_complexity(self, binary: np.ndarray) -> float:
        """Calculate complexity based on line intersections."""
        try:
            # Detect lines using Hough transform
            edges = cv2.Canny(binary, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)

            if lines is None or len(lines) < 2:
                return 0.0

            # Count intersections
            intersection_count = 0
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]

                    # Check if lines intersect
                    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    if abs(denom) > 1e-10:
                        t = ((x1 - x3) * (y3 - y4) -
                             (y1 - y3) * (x3 - x4)) / denom
                        u = -((x1 - x2) * (y1 - y3) -
                              (y1 - y2) * (x1 - x3)) / denom

                        if 0 <= t <= 1 and 0 <= u <= 1:
                            intersection_count += 1

            # Normalize by number of lines
            max_intersections = len(lines) * (len(lines) - 1) / 2
            if max_intersections > 0:
                return min(1.0, intersection_count / max_intersections)

            return 0.0

        except Exception as e:
            logger.warning(
                f"Error calculating line intersection complexity: {e}")
            return 0.0

    def _calculate_curve_complexity(self, binary: np.ndarray) -> float:
        """Calculate complexity based on curves and their properties."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            # Analyze curve properties
            total_curves = 0
            total_complexity = 0.0

            for contour in contours:
                if len(contour) < 5:
                    continue

                # Fit ellipse if possible
                if len(contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        # Ellipse complexity based on eccentricity
                        (center, (width, height), angle) = ellipse
                        if width > 0 and height > 0:
                            eccentricity = abs(
                                width - height) / max(width, height)
                            total_complexity += eccentricity
                            total_curves += 1
                    except:
                        pass

                # Calculate contour complexity
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if area > 0:
                    # Circularity measure
                    circularity = 4 * math.pi * area / (perimeter ** 2)
                    total_complexity += (1 - circularity)
                    total_curves += 1

            if total_curves > 0:
                return np.clip(total_complexity / total_curves, 0.0, 1.0)

            return 0.0

        except Exception as e:
            logger.warning(f"Error calculating curve complexity: {e}")
            return 0.0

    def _calculate_node_density_complexity(self, binary: np.ndarray) -> float:
        """Calculate complexity based on node density (junction points)."""
        try:
            # Skeletonize the pattern
            from skimage.morphology import skeletonize
            skeleton = skeletonize(binary > 0)

            # Find junction points (nodes with 3+ neighbors)
            kernel = np.ones((3, 3), np.uint8)
            kernel[1, 1] = 0

            # Count neighbors for each skeleton pixel
            neighbor_count = cv2.filter2D(
                skeleton.astype(np.uint8), -1, kernel)

            # Junction points have 3 or more neighbors
            junctions = (skeleton > 0) & (neighbor_count >= 3)
            junction_count = np.sum(junctions)

            # Calculate density
            skeleton_pixels = np.sum(skeleton > 0)
            if skeleton_pixels > 0:
                density = junction_count / skeleton_pixels
                # Scale up for normalization
                return np.clip(density * 5, 0.0, 1.0)

            return 0.0

        except Exception as e:
            logger.warning(f"Error calculating node density complexity: {e}")
            return 0.0

    def extract_comprehensive_features(self, image: np.ndarray) -> Dict:
        """
        Extract comprehensive mathematical and symmetry features.

        Args:
            image: Input kolam pattern image

        Returns:
            Dictionary of all extracted features
        """
        logger.info("Extracting comprehensive mathematical features...")

        # Ensure binary image for analysis
        binary = self._ensure_binary(image)

        # Symmetry analysis
        symmetries = self.analyze_pattern_symmetries(binary)

        # Mathematical properties
        math_props = self.calculate_mathematical_properties(binary)

        # Combine all features
        comprehensive_features = {
            'symmetries': {
                sym_type.value: {
                    'confidence': analysis.confidence_score,
                    'order': analysis.symmetry_order,
                    'angle': analysis.symmetry_angle,
                    'center': analysis.symmetry_center,
                    'strength': analysis.symmetry_strength
                }
                for sym_type, analysis in symmetries.items()
            },
            'mathematical_properties': {
                'fractal_dimension': math_props.fractal_dimension,
                'lacunarity': math_props.lacunarity,
                'correlation_dimension': math_props.correlation_dimension,
                'lyapunov_exponent': math_props.lyapunov_exponent,
                'tessellation_order': math_props.tessellation_order,
                'grid_complexity': math_props.grid_complexity,
                'motif_count': math_props.motif_count,
                'connectivity_index': math_props.connectivity_index,
                'curvature_measure': math_props.curvature_measure,
                'aspect_ratio': math_props.aspect_ratio
            },
            'dominant_symmetries': self._identify_dominant_symmetries(symmetries),
            'pattern_complexity_score': self._calculate_complexity_score(symmetries, math_props, binary),
            'extraction_timestamp': datetime.now().isoformat(),
            'analysis_version': '2.0'
        }

        logger.info("Comprehensive feature extraction completed")
        return comprehensive_features

    def _identify_dominant_symmetries(self, symmetries: Dict) -> List[str]:
        """Identify the most significant symmetries."""
        if not symmetries:
            return []

        # Sort by confidence score
        sorted_symmetries = sorted(
            symmetries.items(),
            key=lambda x: x[1].confidence_score,
            reverse=True
        )

        # Return top symmetries above threshold
        dominant = []
        for sym_type, analysis in sorted_symmetries:
            if analysis.confidence_score >= self.min_symmetry_strength:
                dominant.append(sym_type.value)

        return dominant[:3]  # Top 3 symmetries

    def _calculate_complexity_score(self, symmetries: Dict, math_props: MathematicalProperties, binary: np.ndarray) -> float:
        """Calculate overall pattern complexity score."""
        try:
            # Base complexity from fractal dimension
            # Normalize to [0, 1]
            base_complexity = (math_props.fractal_dimension - 1.0)

            # Add symmetry complexity
            symmetry_bonus = min(len(symmetries) * 0.1, 0.3)

            # Add motif complexity
            motif_bonus = min(math_props.motif_count / 20.0, 0.3)

            # Calculate line intersection complexity
            line_intersection_score = self._calculate_line_intersection_complexity(
                binary)

            # Calculate curve complexity
            curve_complexity_score = self._calculate_curve_complexity(binary)

            # Calculate node density complexity
            node_density_score = self._calculate_node_density_complexity(
                binary)

            # Combine all factors with weights
            total_complexity = (
                base_complexity * 0.2 +
                symmetry_bonus * 0.2 +
                motif_bonus * 0.15 +
                line_intersection_score * 0.2 +
                curve_complexity_score * 0.15 +
                node_density_score * 0.1
            )

            return np.clip(total_complexity, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating complexity score: {e}")
            return 0.5


def main():
    """Demonstrate the symmetry analyzer."""
    # Initialize analyzer
    analyzer = KolamSymmetryAnalyzer()

    # Create a test pattern (simple symmetric pattern)
    test_size = 200
    test_pattern = np.zeros((test_size, test_size), dtype=np.uint8)

    # Create a simple 4-fold rotational symmetry pattern
    center = test_size // 2
    for y in range(test_size):
        for x in range(test_size):
            # Distance from center
            dy = y - center
            dx = x - center
            distance = math.sqrt(dx*dx + dy*dy)

            # Create radial pattern
            angle = math.atan2(dy, dx)
            if distance < 80 and abs(math.sin(4 * angle)) < 0.3:  # 4-fold symmetry
                test_pattern[y, x] = 255

    print("KOLAM SYMMETRY ANALYZER DEMONSTRATION")
    print("="*50)

    # Analyze symmetries
    symmetries = analyzer.analyze_pattern_symmetries(test_pattern)

    print(f"\nSymmetries detected: {len(symmetries)}")
    for sym_type, analysis in symmetries.items():
        print(f"  • {sym_type.value}: {analysis.confidence_score:.3f} confidence")

    # Calculate mathematical properties
    math_props = analyzer.calculate_mathematical_properties(test_pattern)

    print("\nMathematical Properties:")
    print(f"  • Fractal Dimension: {math_props.fractal_dimension:.3f}")
    print(f"  • Lacunarity: {math_props.lacunarity:.3f}")
    print(f"  • Correlation Dimension: {math_props.correlation_dimension:.3f}")
    print(f"  • Motif Count: {math_props.motif_count}")
    print(f"  • Connectivity Index: {math_props.connectivity_index:.3f}")
    print(f"  • Grid Complexity: {math_props.grid_complexity:.3f}")

    # Extract comprehensive features
    features = analyzer.extract_comprehensive_features(test_pattern)

    print(
        f"\nDominant Symmetries: {', '.join(features['dominant_symmetries'])}")
    print(
        f"Pattern Complexity Score: {features['pattern_complexity_score']:.3f}")

    print("\nSymmetry analyzer is ready for advanced kolam pattern analysis!")


if __name__ == "__main__":
    main()
