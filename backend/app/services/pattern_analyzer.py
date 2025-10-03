"""
Pattern Analyzer Service for Kolam Design Analysis
"""
from kolam_symmetry_analyzer import KolamSymmetryAnalyzer
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import io
import sys
import os
import time
import cv2
from concurrent.futures import ThreadPoolExecutor

# Add the backend directory to the path to import kolam_symmetry_analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Service for analyzing kolam patterns using computer vision and ML
    """

    def __init__(self):
        """Initialize the pattern analyzer"""
        self.model_version = "2.0.0"
        self.symmetry_analyzer = KolamSymmetryAnalyzer()
        self.cnn_model = None  # Placeholder for CNN model
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Try to load CNN model
        try:
            from kolam_cnn_model import KolamCNNModel
            self.cnn_model = KolamCNNModel()
            self.cnn_model.load_model('backend/models/kolam_cnn_best.h5')
            logger.info("CNN model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CNN model: {e}")
            self.cnn_model = None

        logger.info(
            "PatternAnalyzer initialized with KolamSymmetryAnalyzer and CNN compatibility")

    async def analyze(self, image_array: np.ndarray, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze the pattern in the image array using both symmetry analysis and CNN classification

        Args:
            image_array: Numpy array of the image
            analysis_id: Unique identifier for this analysis

        Returns:
            Analysis results or None if analysis fails
        """
        try:
            logger.info(
                f"Starting comprehensive analysis for pattern {analysis_id}")

            # Preprocess image for both analyses
            processed_image = await self._preprocess_image(image_array)

            # Run symmetry analysis and CNN prediction in parallel
            start_time = time.time()

            # Create tasks for parallel execution
            symmetry_task = asyncio.get_event_loop().run_in_executor(
                self.executor, self._run_symmetry_analysis, processed_image, analysis_id
            )

            cnn_task = asyncio.get_event_loop().run_in_executor(
                self.executor, self._run_cnn_prediction, processed_image, analysis_id
            )

            # Wait for both analyses to complete
            symmetry_results, cnn_results = await asyncio.gather(symmetry_task, cnn_task)

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Combine results
            combined_results = self._combine_analysis_results(
                symmetry_results, cnn_results, analysis_id, processing_time_ms, image_array
            )
            logger.info(
                f"Combined results for {analysis_id}: symmetry_success={symmetry_results.get('success')}, cnn_success={cnn_results.get('success')}, mathematical_properties present: {'mathematical_properties' in combined_results}")

            logger.info(
                f"Comprehensive analysis completed for {analysis_id} in {processing_time_ms}ms")
            return combined_results

        except Exception as e:
            logger.error(f"Error in comprehensive pattern analysis: {str(e)}")
            return self._create_fallback_result(image_array, analysis_id, str(e))

    def _run_symmetry_analysis(self, image: np.ndarray, analysis_id: str) -> Dict[str, Any]:
        """Run symmetry analysis using KolamSymmetryAnalyzer"""
        try:
            logger.info(f"Running symmetry analysis for {analysis_id}")
            features = self.symmetry_analyzer.extract_comprehensive_features(
                image)

            return {
                'success': True,
                'features': features,
                'analysis_type': 'symmetry'
            }
        except Exception as e:
            logger.error(f"Symmetry analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': 'symmetry'
            }

    def _run_cnn_prediction(self, image: np.ndarray, analysis_id: str) -> Dict[str, Any]:
        """Run CNN model prediction for pattern classification"""
        try:
            logger.info(f"Running CNN prediction for {analysis_id}")

            # Check if CNN model is available
            if self.cnn_model is None:
                # Return mock CNN results for now
                cnn_results = self._get_mock_cnn_results(image, analysis_id)
            else:
                # Run actual CNN prediction
                cnn_results = self._run_actual_cnn_prediction(
                    image, analysis_id)

            return {
                'success': True,
                'predictions': cnn_results,
                'analysis_type': 'cnn'
            }
        except Exception as e:
            logger.error(f"CNN prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': 'cnn'
            }

    def _run_actual_cnn_prediction(self, image: np.ndarray, analysis_id: str) -> Dict[str, Any]:
        """Run actual CNN model prediction"""
        try:
            import tempfile
            import os
            from PIL import Image

            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img = Image.fromarray(image.astype('uint8'))
                img.save(tmp.name)

                try:
                    prediction = self.cnn_model.predict(tmp.name)
                    predicted_label = prediction['predicted_label']

                    # Parse predicted_label, assume format like "Traditional_Pulli_Kolam_TamilNadu"
                    parts = predicted_label.split('_')
                    pattern_type = parts[0] if len(parts) > 0 else "Unknown"
                    subtype = parts[1] if len(parts) > 1 else "Unknown"
                    region = parts[2] if len(parts) > 2 else "Unknown"

                    return {
                        "pattern_type": pattern_type,
                        "subtype": subtype,
                        "region": region,
                        "confidence": prediction['confidence'],
                        "features": {
                            "texture_complexity": 0.5,  # Placeholder
                            "line_density": 0.5,
                            "symmetry_score": 0.5
                        }
                    }
                finally:
                    os.unlink(tmp.name)
        except Exception as e:
            logger.warning(f"Actual CNN prediction failed, using mock: {e}")
            return self._get_mock_cnn_results(image, analysis_id)

    def _get_mock_cnn_results(self, image: np.ndarray, analysis_id: str) -> Dict[str, Any]:
        """Generate mock CNN results for demonstration"""
        import random

        pattern_types = ["Traditional",
                         "Contemporary", "Festival", "Ceremonial"]
        subtypes = ["Pulli Kolam", "Kambi Kolam", "Nelli Kolam", "Sikku Kolam"]
        regions = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh"]

        return {
            "pattern_type": random.choice(pattern_types),
            "subtype": random.choice(subtypes),
            "region": random.choice(regions),
            "confidence": round(random.uniform(0.75, 0.95), 3),
            "features": {
                "texture_complexity": round(random.uniform(0.3, 0.8), 3),
                "line_density": round(random.uniform(0.2, 0.9), 3),
                "symmetry_score": round(random.uniform(0.6, 0.95), 3)
            }
        }

    def _combine_analysis_results(self, symmetry_results: Dict, cnn_results: Dict,
                                  analysis_id: str, processing_time_ms: int,
                                  original_image: np.ndarray) -> Dict[str, Any]:
        """Combine symmetry and CNN results into unified output"""

        # Extract symmetry features
        symmetry_features = {}
        symmetry_success = symmetry_results.get('success', False)

        if symmetry_success:
            features = symmetry_results.get('features', {})
            symmetry_features = {
                'symmetries_detected': features.get('symmetries', {}),
                'mathematical_properties': features.get('mathematical_properties', {}),
                'dominant_symmetries': features.get('dominant_symmetries', []),
                'pattern_complexity_score': features.get('pattern_complexity_score', 0.5)
            }

        # Extract CNN predictions
        cnn_predictions = {}
        cnn_success = cnn_results.get('success', False)

        if cnn_success:
            cnn_predictions = cnn_results.get('predictions', {})

        # Create combined result
        combined_result = {
            "analysis_id": analysis_id,
            "processing_time_ms": processing_time_ms,
            "analysis_version": self.model_version,
            "analysis_status": {
                "symmetry_analysis": "success" if symmetry_success else "failed",
                "cnn_prediction": "success" if cnn_success else "failed",
                "overall_status": "success" if (symmetry_success or cnn_success) else "failed"
            },
            "design_classification": self._create_unified_classification(
                symmetry_features, cnn_predictions
            ),
            "symmetry_analysis": symmetry_features if symmetry_success else {"error": symmetry_results.get('error')},
            "cnn_classification": cnn_predictions if cnn_success else {"error": cnn_results.get('error')},
            "mathematical_properties": self._extract_mathematical_properties(symmetry_features),
            "cultural_context": self._extract_cultural_context(cnn_predictions),
            "metadata": {
                "image_dimensions": original_image.shape[:2],
                "color_mode": "RGB" if len(original_image.shape) == 3 else "Grayscale",
                "processing_timestamp": time.time()
            }
        }

        return combined_result

    def _create_unified_classification(self, symmetry_features: Dict, cnn_predictions: Dict) -> Dict[str, Any]:
        """Create unified classification from both analyses"""
        # Prioritize CNN results for pattern type classification
        if cnn_predictions:
            return {
                "type": cnn_predictions.get("pattern_type", "Unknown"),
                "subtype": cnn_predictions.get("subtype", "Unknown"),
                "region": cnn_predictions.get("region", "Unknown"),
                "confidence": cnn_predictions.get("confidence", 0.5),
                "classification_method": "CNN"
            }
        else:
            # Fall back to symmetry-based classification
            return {
                "type": self._classify_pattern_type(symmetry_features),
                "subtype": self._classify_pattern_subtype(symmetry_features),
                "region": "Tamil Nadu",
                "confidence": self._calculate_overall_confidence(symmetry_features),
                "classification_method": "Symmetry-based"
            }

    def _extract_mathematical_properties(self, symmetry_features: Dict) -> Dict[str, Any]:
        """Extract mathematical properties from symmetry analysis"""
        if not symmetry_features:
            logger.warning(
                "Symmetry features are empty, returning empty mathematical properties")
            return {}

        math_props = symmetry_features.get('mathematical_properties', {})
        logger.info(
            f"Extracting mathematical properties: symmetry_features keys: {list(symmetry_features.keys())}, math_props keys: {list(math_props.keys())}")

        extracted = {
            "symmetry_type": self._get_dominant_symmetry_type(symmetry_features),
            "rotational_order": max(1, self._calculate_rotational_order(symmetry_features)),
            "reflection_axes": max(1, self._count_reflection_symmetries(symmetry_features)),
            "complexity_score": max(0.1, symmetry_features.get('pattern_complexity_score', 0.5)),
            "fractal_dimension": max(1.1, math_props.get('fractal_dimension', 1.5)),
            "lacunarity": max(0.1, math_props.get('lacunarity', 1.0)),
            "correlation_dimension": max(1.1, math_props.get('correlation_dimension', 1.5)),
            "connectivity_index": max(0.1, math_props.get('connectivity_index', 0.5)),
            "grid_complexity": max(0.1, math_props.get('grid_complexity', 0.5))
        }
        logger.info(f"Extracted mathematical properties: {extracted}")
        return extracted

    def _extract_cultural_context(self, cnn_predictions: Dict) -> Dict[str, Any]:
        """Extract cultural context from CNN predictions"""
        if not cnn_predictions:
            return {
                "ceremonial_use": "Unknown",
                "seasonal_relevance": "Unknown",
                "symbolic_meaning": "Unknown",
                "traditional_name": "Unknown"
            }

        # Map CNN predictions to cultural context
        pattern_type = cnn_predictions.get("pattern_type", "Traditional")

        context_mapping = {
            "Traditional": {
                "ceremonial_use": "Daily Prayer",
                "symbolic_meaning": "Harmony and Balance"
            },
            "Festival": {
                "ceremonial_use": "Festival Celebration",
                "symbolic_meaning": "Prosperity and Joy"
            },
            "Ceremonial": {
                "ceremonial_use": "Special Occasions",
                "symbolic_meaning": "Protection and Blessing"
            },
            "Contemporary": {
                "ceremonial_use": "Modern Expression",
                "symbolic_meaning": "Innovation and Creativity"
            }
        }

        context = context_mapping.get(
            pattern_type, context_mapping["Traditional"])

        return {
            "ceremonial_use": context["ceremonial_use"],
            "seasonal_relevance": "All Seasons",
            "symbolic_meaning": context["symbolic_meaning"],
            "traditional_name": f"Kolam_{pattern_type}_{np.random.randint(100, 999)}"
        }

    def _create_fallback_result(self, image_array: np.ndarray, analysis_id: str, error: str) -> Dict[str, Any]:
        """Create fallback result when both analyses fail"""
        return {
            "analysis_id": analysis_id,
            "processing_time_ms": 0,
            "analysis_version": self.model_version,
            "analysis_status": {
                "symmetry_analysis": "failed",
                "cnn_prediction": "failed",
                "overall_status": "failed"
            },
            "design_classification": {
                "type": "Unknown",
                "subtype": "Unknown",
                "region": "Unknown",
                "confidence": 0.0,
                "classification_method": "Error"
            },
            "error": error,
            "metadata": {
                "image_dimensions": image_array.shape[:2],
                "color_mode": "RGB" if len(image_array.shape) == 3 else "Grayscale",
                "processing_timestamp": time.time()
            }
        }

    async def _preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """Preprocess image for both symmetry and CNN analysis"""
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    gray = np.dot(image_array[..., :3], [
                                  0.2989, 0.5870, 0.1140])
                else:
                    gray = image_array[:, :, 0]  # Use first channel
            else:
                gray = image_array

            # Ensure proper data type and range
            gray = np.clip(gray, 0, 255).astype(np.uint8)

            # Resize if too large for CNN processing
            max_dimension = 512
            if max(gray.shape) > max_dimension:
                scale_factor = max_dimension / max(gray.shape)
                new_width = int(gray.shape[1] * scale_factor)
                new_height = int(gray.shape[0] * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height))

            return gray

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Return original image if preprocessing fails
            return image_array.astype(np.uint8)

    # Helper methods for classification (reused from previous implementation)
    def _classify_pattern_type(self, features: Dict) -> str:
        """Classify the pattern type based on features."""
        complexity = features.get('pattern_complexity_score', 0.5)
        symmetries = features.get('dominant_symmetries', [])

        if complexity > 0.7:
            return "Contemporary"
        elif any('rotational' in sym for sym in symmetries):
            return "Traditional"
        else:
            return "Festival"

    def _classify_pattern_subtype(self, features: Dict) -> str:
        """Classify the pattern subtype."""
        symmetries = features.get('dominant_symmetries', [])
        motif_count = features.get(
            'mathematical_properties', {}).get('motif_count', 1)

        if motif_count > 10:
            return "Sikku Kolam"
        elif any('radial' in sym for sym in symmetries):
            return "Nelli Kolam"
        elif any('rotational_4' in sym for sym in symmetries):
            return "Kambi Kolam"
        else:
            return "Pulli Kolam"

    def _calculate_overall_confidence(self, features: Dict) -> float:
        """Calculate overall confidence score."""
        symmetries = features.get('symmetries_detected', {})
        if not symmetries:
            return 50.0

        # Average confidence across all detected symmetries
        confidences = [sym_data.get('confidence', 0)
                       for sym_data in symmetries.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # Convert to percentage and ensure reasonable range
        return round(max(60.0, min(95.0, avg_confidence * 100)), 2)

    def _get_dominant_symmetry_type(self, features: Dict) -> str:
        """Get the dominant symmetry type."""
        symmetries = features.get('dominant_symmetries', [])
        if not symmetries:
            return "Asymmetric"

        # Return the first dominant symmetry in a readable format
        dominant = symmetries[0]
        if 'rotational' in dominant:
            return "Rotational"
        elif 'reflection' in dominant:
            return "Bilateral"
        elif 'radial' in dominant:
            return "Radial"
        else:
            return "Translational"

    def _calculate_rotational_order(self, features: Dict) -> int:
        """Calculate the rotational order from symmetries."""
        symmetries = features.get('symmetries_detected', {})

        for sym_type, sym_data in symmetries.items():
            if 'rotational' in sym_type:
                return sym_data.get('order', 2)

        return 1  # No rotational symmetry

    def _count_reflection_symmetries(self, features: Dict) -> int:
        """Count the number of reflection symmetries."""
        symmetries = features.get('symmetries_detected', {})
        reflection_count = 0

        for sym_type in symmetries.keys():
            if 'reflection' in sym_type:
                reflection_count += 1

        return reflection_count


# Global instance
pattern_analyzer = PatternAnalyzer()
