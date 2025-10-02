"""
Kolam Advanced Symmetry Analysis and Mathematical Property Extraction

This module performs comprehensive mathematical analysis of kolam patterns including
symmetry detection, fractal dimension calculation, tessellation analysis, and
advanced geometric property extraction for intelligent pattern regeneration.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, transform, filters, feature
from skimage.transform import rotate, rescale
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

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
        if image.max() > 1:
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
            flipped = binary.T
            center = (height // 2, width // 2)
        else:
            raise ValueError(f"Unknown axis: {axis}")

        # Calculate similarity
        difference = np.abs(binary.astype(np.float32) -
                            flipped.astype(np.float32))
        similarity = 1.0 - (np.mean(difference) / 255.0)

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
                zip(violation_points[0], violation_points[1]))
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
            rotated = rotate(binary, angle_i, center=(
                center_x, center_y), preserve_range=True)
            rotated = (rotated > 127).astype(np.uint8) * 255
            rotated_images.append(rotated)

        # Compare all rotations with original
        similarities = []
        for rotated in rotated_images:
            difference = np.abs(binary.astype(
                np.float32) - rotated.astype(np.float32))
            similarity = 1.0 - (np.mean(difference) / 255.0)
            similarities.append(similarity)

        # Average similarity across all rotations
        avg_similarity = np.mean(similarities)

        # Find consistent violation points across rotations
        violation_points = []
        for y in range(height):
            for x in range(width):
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
            violation_points=violation_points
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
                binary.shape, center_y, center_x, angle)

            # Compare with original
            difference = np.abs(binary.astype(np.float32) -
                                radial_test.astype(np.float32))
            similarity = 1.0 - (np.mean(difference) / 255.0)
            similarities.append(similarity)

        avg_similarity = np.mean(similarities)

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

# In backend/kolam_symmetry_analyzer.py

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
        # Use autocorrelation to detect periodic patterns
        try:
            # Calculate 2D autocorrelation
            autocorr = self._calculate_autocorrelation(binary)

            # Find peaks in autocorrelation (indicating repetition)
            peaks = self._find_autocorrelation_peaks(autocorr)

            if len(peaks) >= 2:
                # Calculate average translation vector
                vectors = []
                for i in range(1, len(peaks)):
                    prev_peak = peaks[i-1]
                    curr_peak = peaks[i]
                    vector = (curr_peak[0] - prev_peak[0],
                              curr_peak[1] - prev_peak[1])
                    vectors.append(vector)

                if vectors:
                    avg_vector = np.mean(vectors, axis=0)

                    # Calculate confidence from peak strengths
                    # Normalize autocorrelation for peak strength calculation
                    # Normalize autocorrelation for peak strength calculation
                    autocorr_range = autocorr.max() - autocorr.min()
                    if autocorr_range > 0:
                        autocorr_norm = (
                            autocorr - autocorr.min()) / autocorr_range
                    else:
                        autocorr_norm = np.zeros_like(autocorr)

                    # Calculate peak strengths from normalized autocorrelation
                    peak_strengths = [autocorr_norm[p[0], p[1]] for p in peaks]
                    confidence = np.mean(peak_strengths)

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
        # Normalize image
        binary_float = binary.astype(np.float32) / 255.0

        # Calculate autocorrelation using FFT
        fft = np.fft.fft2(binary_float)
        autocorr_fft = fft * np.conj(fft)
        autocorr = np.fft.ifft2(autocorr_fft).real

        return autocorr

    def _find_autocorrelation_peaks(self, autocorr: np.ndarray) -> List[Tuple[int, int]]:
        """Find significant peaks in autocorrelation."""
        # Normalize autocorrelation
        autocorr_norm = (autocorr - autocorr.min()) / \
            (autocorr.max() - autocorr.min())

        # Find local maxima above threshold
        peaks = []
        threshold = 0.7

        for y in range(1, autocorr.shape[0] - 1):
            for x in range(1, autocorr.shape[1] - 1):
                if autocorr_norm[y, x] > threshold:
                    # Check if local maximum
                    neighborhood = autocorr_norm[y-1:y+2, x-1:x+2]
                    if autocorr_norm[y, x] == neighborhood.max():
                        peaks.append((y, x))

        return peaks

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
            binary_float = binary.astype(np.float32)
            sizes = []
            counts = []

            for box_size in self.fractal_box_sizes:
                if box_size > min(binary.shape):
                    continue

                # Count non-empty boxes
                count = 0
                for y in range(0, binary.shape[0], box_size):
                    for x in range(0, binary.shape[1], box_size):
                        box = binary_float[y:y+box_size, x:x+box_size]
                        if np.sum(box) > 0:
                            count += 1

                if count > 0:
                    sizes.append(box_size)
                    counts.append(count)

            if len(sizes) >= 2:
                # Linear regression in log-log space
                log_sizes = np.log(1.0 / np.array(sizes))
                log_counts = np.log(np.array(counts))

                # Simple linear fit
                coeffs = np.polyfit(log_sizes, log_counts, 1)
                # Negative slope in log-log space
                fractal_dimension = -coeffs[0]

                # Clamp to reasonable range
                return max(1.0, min(2.0, fractal_dimension))

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
                        mass = np.sum(box)
                        masses.append(mass)

                if masses:
                    masses = np.array(masses)
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

            # Calculate pairwise distances
            distances = []
            for i in range(len(y_coords)):
                for j in range(i + 1, len(y_coords)):
                    dist = math.sqrt(
                        (y_coords[i] - y_coords[j])**2 + (x_coords[i] - x_coords[j])**2)
                    distances.append(dist)

            if not distances:
                return 1.0

            distances = np.array(distances)

            # Correlation integral for different radii
            radii = np.logspace(0, 2, 10)  # Log-spaced radii
            correlation_values = []

            for r in radii:
                # Count pairs within radius r
                within_r = np.sum(distances <= r)
                total_pairs = len(distances)
                if total_pairs > 0:
                    correlation = within_r / total_pairs
                    correlation_values.append(correlation)

            if len(correlation_values) >= 2:
                # Fit power law: C(r) ~ r^D
                log_r = np.log(radii[:len(correlation_values)])
                log_c = np.log(np.array(correlation_values) + 1e-10)

                coeffs = np.polyfit(log_r, log_c, 1)
                return coeffs[0]  # Slope gives correlation dimension

        except Exception as e:
            logger.warning(f"Error calculating correlation dimension: {e}")

        return 1.5  # Default value

    def _calculate_lyapunov_exponent(self, binary: np.ndarray) -> float:
        """Calculate Lyapunov exponent (measure of chaos/sensitivity)."""
        try:
            # Simplified Lyapunov exponent calculation
            # For kolam patterns, this measures how sensitive the pattern is to small changes

            # Create slightly perturbed version
            noise = np.random.random(binary.shape) * 0.1
            perturbed = np.clip(binary.astype(np.float32) + noise, 0, 1)

            # Calculate difference
            difference = np.abs(binary.astype(np.float32) - perturbed)
            avg_difference = np.mean(difference)

            # Lyapunov exponent approximation
            lyapunov_exp = - \
                math.log(1 - avg_difference) if avg_difference < 1 else 1.0

            return min(1.0, lyapunov_exp)

        except Exception as e:
            logger.warning(f"Error calculating Lyapunov exponent: {e}")
            return 0.1  # Default small positive value

    def _analyze_tessellation(self, binary: np.ndarray) -> int:
        """Analyze tessellation properties."""
        try:
            # Use Voronoi tessellation analysis
            points = np.where(binary > 0)
            if len(points[0]) < 5:
                return 1

            # Simple tessellation order estimation
            # Count connected components
            labeled, num_components = measure.label(binary, return_num=True)

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
            edges = cv2.Canny(binary.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / \
                (edges.shape[0] * edges.shape[1])

            # Calculate line complexity
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 30, minLineLength=10, maxLineGap=5)

            if lines is not None:
                line_count = len(lines)
                avg_line_length = np.mean(
                    [math.sqrt((x2-x1)**2 + (y2-y1)**2) for line in lines for x1, y1, x2, y2 in line])
            else:
                line_count = 0
                avg_line_length = 0

            # Combine metrics
            complexity = (edge_density * 0.5) + (min(line_count /
                                                     100, 1.0) * 0.3) + (avg_line_length / 50 * 0.2)

            return min(1.0, complexity)

        except Exception as e:
            logger.warning(f"Error calculating grid complexity: {e}")
            return 0.5

    def _count_motifs(self, binary: np.ndarray) -> int:
        """Count distinct motifs in the pattern."""
        try:
            # Label connected components
            labeled, num_components = measure.label(binary, return_num=True)

            # Filter out very small components (noise)
            min_size = 10
            significant_components = 0

            for i in range(1, num_components + 1):
                component = (labeled == i)
                if np.sum(component) >= min_size:
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
            eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1)

            # Calculate connectivity as ratio of eroded to original
            original_mass = np.sum(binary)
            eroded_mass = np.sum(eroded)

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
            contours, _ = cv2.findContours(binary.astype(
                np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

                    # Calculate angles
                    v1 = p1 - p2
                    v2 = p3 - p2

                    # Avoid division by zero
                    norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    norm2 = math.sqrt(v2[0]**2 + v2[1]**2)

                    if norm1 > 0 and norm2 > 0:
                        cos_angle = (v1[0]*v2[0] + v1[1] *
                                     v2[1]) / (norm1 * norm2)
                        angle = math.acos(max(-1, min(1, cos_angle)))
                        curvature_sum += abs(angle)

                total_curvature += curvature_sum
                total_length += len(contour)

            if total_length > 0:
                avg_curvature = total_curvature / total_length
                return min(1.0, avg_curvature / math.pi)  # Normalize

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

    def extract_comprehensive_features(self, image: np.ndarray) -> Dict:
        """
        Extract comprehensive mathematical and symmetry features.

        Args:
            image: Input kolam pattern image

        Returns:
            Dictionary of all extracted features
        """
        logger.info("Extracting comprehensive mathematical features...")

        # Symmetry analysis
        symmetries = self.analyze_pattern_symmetries(image)

        # Mathematical properties
        math_props = self.calculate_mathematical_properties(image)

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
            'pattern_complexity_score': self._calculate_complexity_score(symmetries, math_props),
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

    def _calculate_complexity_score(self, symmetries: Dict, math_props: MathematicalProperties) -> float:
        """Calculate overall pattern complexity score."""
        # Base complexity from fractal dimension
        base_complexity = (math_props.fractal_dimension -
                           1.0) / 1.0  # Normalize to [0, 1]

        # Add symmetry complexity
        symmetry_bonus = len(symmetries) * 0.1

        # Add motif complexity
        motif_bonus = min(math_props.motif_count / 20.0, 0.3)

        # Combine factors
        total_complexity = base_complexity + symmetry_bonus + motif_bonus

        return min(1.0, total_complexity)


def main():
    """Demonstrate the symmetry analyzer."""
    # Initialize analyzer
    analyzer = KolamSymmetryAnalyzer()

    # Create a test pattern (simple symmetric pattern)
    test_size = 100
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
            if distance < 40 and abs(math.sin(4 * angle)) < 0.3:  # 4-fold symmetry
                test_pattern[y, x] = 255

    print("KOLAM SYMMETRY ANALYZER DEMONSTRATION")
    print("="*50)

    # Analyze symmetries
    symmetries = analyzer.analyze_pattern_symmetries(test_pattern)

    print(f"Symmetries detected: {len(symmetries)}")
    for sym_type, analysis in symmetries.items():
        print(f"- {sym_type.value}: {analysis.confidence_score:.3f} confidence")

    # Calculate mathematical properties
    math_props = analyzer.calculate_mathematical_properties(test_pattern)

    print("\nMathematical Properties:")
    print(f"- Fractal Dimension: {math_props.fractal_dimension:.3f}")
    print(f"- Lacunarity: {math_props.lacunarity:.3f}")
    print(f"- Motif Count: {math_props.motif_count}")
    print(f"- Connectivity Index: {math_props.connectivity_index:.3f}")

    # Extract comprehensive features
    features = analyzer.extract_comprehensive_features(test_pattern)

    print(f"\nDominant Symmetries: {features['dominant_symmetries']}")
    print(
        f"Pattern Complexity Score: {features['pattern_complexity_score']:.3f}")

    print("\nSymmetry analyzer is ready for advanced kolam pattern analysis!")


if __name__ == "__main__":
    main()
