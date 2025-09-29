"""
Kolam AI API Endpoints

This module provides FastAPI endpoints for kolam pattern recognition and generation
using the trained CNN and GAN models.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional
import os
import uuid
from datetime import datetime
import logging
from PIL import Image
import io

# Import our custom modules
from kolam_cnn_model import KolamCNNModel
from kolam_gan_model import KolamGAN
from kolam_ai_pipeline import KolamAIPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global model instances (loaded once at startup)
cnn_model = None
gan_model = None
ai_pipeline = None


def load_models():
    """
    Load the trained models at startup.
    """
    global cnn_model, gan_model, ai_pipeline

    try:
        # Load CNN model
        cnn_model = KolamCNNModel(img_height=224, img_width=224)
        cnn_model.load_model('backend/models/kolam_cnn_final.h5')
        logger.info("CNN model loaded successfully")

        # Load GAN model
        gan_model = KolamGAN(img_height=128, img_width=128)
        gan_model.load_models()
        logger.info("GAN model loaded successfully")

        # Load AI pipeline
        ai_pipeline = KolamAIPipeline(img_height=128, img_width=128)
        ai_pipeline.load_training_history()
        logger.info("AI pipeline loaded successfully")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def preprocess_image(image_file: UploadFile, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess uploaded image for model inference.

    Args:
        image_file: Uploaded image file
        target_size: Target size for the image

    Returns:
        Preprocessed image array
    """
    try:
        # Read image file
        image_data = image_file.file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image
        image = image.resize(target_size)

        # Convert to numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(image)

        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing image: {str(e)}")


@router.post("/recognize", response_model=Dict)
async def recognize_kolam_pattern(image: UploadFile = File(...)):
    """
    Recognize a kolam pattern from an uploaded image.

    Args:
        image: Uploaded image file

    Returns:
        Recognition results with predicted class and confidence
    """
    if cnn_model is None:
        raise HTTPException(status_code=500, detail="CNN model not loaded")

    try:
        # Preprocess image
        img_array = preprocess_image(image, target_size=(224, 224))

        # Make prediction using CNN model
        prediction = cnn_model.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))

        # Get class name (if available)
        class_names = getattr(cnn_model, 'class_names', None)
        predicted_label = class_names[predicted_class] if class_names and predicted_class < len(
            class_names) else f"Class_{predicted_class}"

        result = {
            "predicted_class": int(predicted_class),
            "predicted_label": predicted_label,
            "confidence": confidence,
            "confidence_percentage": confidence * 100,
            "probabilities": prediction[0].tolist(),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            f"Pattern recognized: {predicted_label} (confidence: {confidence:.4f})")
        return result

    except Exception as e:
        logger.error(f"Error in pattern recognition: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Recognition failed: {str(e)}")


@router.post("/generate", response_model=Dict)
async def generate_kolam_patterns(
    num_patterns: int = 1,
    class_condition: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Generate new kolam patterns using the GAN model.

    Args:
        num_patterns: Number of patterns to generate (1-10)
        class_condition: Optional class conditioning (for future use)
        background_tasks: FastAPI background tasks

    Returns:
        Generated pattern information
    """
    if gan_model is None:
        raise HTTPException(status_code=500, detail="GAN model not loaded")

    try:
        # Validate input
        if not 1 <= num_patterns <= 10:
            raise HTTPException(
                status_code=400, detail="num_patterns must be between 1 and 10")

        # Generate patterns
        generated_images = gan_model.generate_kolam(num_patterns)

        if generated_images is None:
            raise HTTPException(
                status_code=500, detail="Pattern generation failed")

        # Create response
        patterns = []
        for i, img in enumerate(generated_images):
            pattern_info = {
                "pattern_id": f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                "image_shape": img.shape,
                "generation_timestamp": datetime.now().isoformat(),
                "generation_method": "gan",
                "class_condition": class_condition
            }
            patterns.append(pattern_info)

        result = {
            "patterns": patterns,
            "num_generated": len(patterns),
            "generation_timestamp": datetime.now().isoformat(),
            "model_info": {
                "generator_params": gan_model.generator.count_params(),
                "latent_dimension": gan_model.latent_dim
            }
        }

        logger.info(f"Generated {len(patterns)} kolam patterns")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pattern generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/analyze_and_generate", response_model=Dict)
async def analyze_and_generate(
    image: UploadFile = File(...),
    num_variations: int = 5,
    background_tasks: BackgroundTasks = None
):
    """
    Complete workflow: analyze input pattern and generate variations.

    Args:
        image: Input kolam image
        num_variations: Number of variations to generate
        background_tasks: FastAPI background tasks

    Returns:
        Complete analysis and generation results
    """
    if ai_pipeline is None:
        raise HTTPException(status_code=500, detail="AI pipeline not loaded")

    try:
        # Save uploaded image temporarily
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        temp_path = f"backend/temp/{temp_filename}"

        os.makedirs("backend/temp", exist_ok=True)

        # Read and save image
        image_data = image.file.read()
        with open(temp_path, "wb") as f:
            f.write(image_data)

        # Use AI pipeline for complete workflow
        workflow_results = ai_pipeline.analyze_and_generate(
            temp_path, num_variations)

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        logger.info("Complete workflow executed successfully")
        return workflow_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze and generate workflow: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Workflow failed: {str(e)}")


@router.get("/model_info", response_model=Dict)
async def get_model_info():
    """
    Get information about the loaded models.

    Returns:
        Model information dictionary
    """
    try:
        info = {
            "cnn_model": {
                "loaded": cnn_model is not None,
                "input_shape": (224, 224, 3) if cnn_model else None,
                "num_classes": cnn_model.num_classes if cnn_model else None
            },
            "gan_model": {
                "loaded": gan_model is not None,
                "generator_shape": (128, 128, 3) if gan_model else None,
                "latent_dimension": gan_model.latent_dim if gan_model else None
            },
            "ai_pipeline": {
                "loaded": ai_pipeline is not None,
                "is_trained": ai_pipeline.is_trained if ai_pipeline else False
            },
            "timestamp": datetime.now().isoformat()
        }

        return info

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the AI models.

    Returns:
        Health status
    """
    try:
        models_loaded = {
            "cnn_model": cnn_model is not None,
            "gan_model": gan_model is not None,
            "ai_pipeline": ai_pipeline is not None
        }

        all_loaded = all(models_loaded.values())

        return {
            "status": "healthy" if all_loaded else "degraded",
            "models_loaded": models_loaded,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Load models on module import
try:
    load_models()
    logger.info("Kolam AI API endpoints initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Kolam AI API: {str(e)}")
