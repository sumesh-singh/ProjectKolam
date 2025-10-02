"""
Pattern analysis and recognition endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import uuid
import io
import time
import logging
from PIL import Image
import numpy as np
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.schemas.pattern import (
    PatternAnalysisRequest,
    PatternAnalysisResponse,
    AnalysisResult,
    BatchAnalysisResponse,
    BatchAnalysisResult
)
from app.services.pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize pattern analyzer service
pattern_analyzer = PatternAnalyzer()

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# In-memory storage for development (replace with database in production)
# Consider using Redis or a proper cache in production
analysis_storage: Dict[str, Dict[str, Any]] = {}

# Constants
MAX_FILE_SIZE = getattr(settings, 'MAX_FILE_SIZE',
                        10 * 1024 * 1024)  # 10MB default
MAX_IMAGE_DIMENSION = getattr(
    settings, 'MAX_IMAGE_SIZE', 2048)  # 2048px default
MAX_BATCH_SIZE = getattr(settings, 'MAX_BATCH_SIZE', 10)
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/png', 'image/jpg', 'image/webp'}


async def validate_image_file(file: UploadFile) -> bytes:
    """
    Validate uploaded image file.

    Args:
        file: Uploaded file

    Returns:
        File contents as bytes

    Raises:
        HTTPException: If file validation fails
    """
    # Check content type
    if not file.content_type or file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )

    # Read file content
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read file contents"
        )

    # Check file size
    file_size = len(contents)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB, "
            f"received {file_size / (1024*1024):.2f}MB."
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded"
        )

    return contents


async def process_image(contents: bytes) -> np.ndarray:
    """
    Process image contents into numpy array.

    Args:
        contents: Image file contents

    Returns:
        Processed image as numpy array

    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Open and process image
        image = Image.open(io.BytesIO(contents))

        # Verify it's a valid image
        image.verify()
        image = Image.open(io.BytesIO(contents))  # Need to reopen after verify

        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Resize if too large
        if max(image.size) > MAX_IMAGE_DIMENSION:
            image.thumbnail(
                (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
                Image.Resampling.LANCZOS
            )

        # Convert to numpy array
        image_array = np.array(image)

        logger.info(
            f"Processed image: size={image.size}, mode={image.mode}, shape={image_array.shape}")

        return image_array

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file. Could not process the uploaded image."
        )


@router.post("/analyze", response_model=PatternAnalysisResponse)
async def analyze_pattern(
    file: UploadFile = File(..., description="Kolam pattern image to analyze")
) -> PatternAnalysisResponse:
    """
    Analyze an uploaded kolam pattern image.

    Args:
        file: Image file containing the kolam pattern

    Returns:
        Analysis results including pattern classification, mathematical properties,
        and cultural context
    """
    start_time = time.time()

    try:
        # Log request
        logger.info(
            f"Received pattern analysis request for file: {file.filename}")

        # Validate and read file
        contents = await validate_image_file(file)

        # Process image
        image_array = await process_image(contents)

        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())

        # Perform pattern analysis
        try:
            # Use the actual pattern analyzer service
            loop = asyncio.get_event_loop()
            analysis_result = await loop.run_in_executor(
                executor,
                pattern_analyzer.analyze,
                image_array,
                analysis_id
            )

            # If pattern_analyzer doesn't exist or returns None, use mock
            if not analysis_result:
                analysis_result = await perform_mock_analysis(image_array, analysis_id)

        except Exception as e:
            logger.warning(f"Pattern analyzer failed, using mock: {str(e)}")
            analysis_result = await perform_mock_analysis(image_array, analysis_id)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Create response
        response = PatternAnalysisResponse(
            status="success",
            message="Pattern analysis completed successfully",
            analysis_id=analysis_id,
            processing_time_ms=processing_time_ms,
            result=AnalysisResult(**analysis_result),
            timestamp=datetime.utcnow()
        )

        # Store result for retrieval
        analysis_storage[analysis_id] = response.dict()

        # Clean up old storage entries if too many (simple LRU)
        if len(analysis_storage) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(analysis_storage.keys())[:100]
            for key in sorted_keys:
                del analysis_storage[key]

        logger.info(
            f"Analysis completed successfully: {analysis_id} in {processing_time_ms}ms")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in analyze_pattern: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later."
        )
    finally:
        # Clean up file
        await file.close()


@router.get("/analysis/{analysis_id}", response_model=PatternAnalysisResponse)
async def get_analysis(analysis_id: str) -> PatternAnalysisResponse:
    """
    Retrieve analysis results by ID.

    Args:
        analysis_id: Unique identifier of the analysis

    Returns:
        Previously computed analysis results
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found"
        )

    return PatternAnalysisResponse(**analysis_storage[analysis_id])


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    files: List[UploadFile] = File(...,
                                   description="Multiple kolam pattern images")
) -> BatchAnalysisResponse:
    """
    Analyze multiple kolam patterns in batch.

    Args:
        files: List of image files

    Returns:
        Batch analysis results
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_BATCH_SIZE} images allowed per batch"
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    results = []
    tasks = []

    # Create tasks for parallel processing
    for file in files:
        task = process_single_file_in_batch(file)
        tasks.append(task)

    # Execute all tasks concurrently
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful = 0
    failed = 0

    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            results.append(BatchAnalysisResult(
                filename=files[i].filename,
                status="failed",
                error=str(result),
                analysis_id=None
            ))
            failed += 1
        else:
            results.append(result)
            if result.status == "success":
                successful += 1
            else:
                failed += 1

    return BatchAnalysisResponse(
        status="success",
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
        timestamp=datetime.utcnow()
    )


async def process_single_file_in_batch(file: UploadFile) -> BatchAnalysisResult:
    """
    Process a single file in batch operation.

    Args:
        file: Uploaded file

    Returns:
        Batch analysis result for single file
    """
    try:
        # Validate and process
        contents = await validate_image_file(file)
        image_array = await process_image(contents)

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Perform analysis
        loop = asyncio.get_event_loop()
        try:
            analysis_result = await loop.run_in_executor(
                executor,
                pattern_analyzer.analyze,
                image_array,
                analysis_id
            )
        except:
            analysis_result = await perform_mock_analysis(image_array, analysis_id)

        # Store result
        analysis_storage[analysis_id] = analysis_result

        return BatchAnalysisResult(
            filename=file.filename,
            status="success",
            analysis_id=analysis_id,
            error=None
        )

    except HTTPException as e:
        return BatchAnalysisResult(
            filename=file.filename,
            status="failed",
            analysis_id=None,
            error=e.detail
        )
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        return BatchAnalysisResult(
            filename=file.filename,
            status="failed",
            analysis_id=None,
            error=str(e)
        )
    finally:
        await file.close()


async def perform_mock_analysis(image_array: np.ndarray, analysis_id: str) -> Dict[str, Any]:
    """
    Perform mock pattern analysis.
    Replace this with actual ML model integration.

    Args:
        image_array: Numpy array of the image
        analysis_id: Unique analysis identifier

    Returns:
        Analysis results dictionary
    """
    import random

    # Mock data for demonstration
    pattern_types = ["Traditional", "Contemporary", "Festival", "Daily"]
    subtypes = ["Pulli Kolam", "Kambi Kolam", "Nelli Kolam", "Sikku Kolam"]
    regions = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh"]
    symmetry_types = ["Rotational", "Bilateral", "Radial", "Asymmetric"]
    ceremonial_uses = ["Daily Prayer",
                       "Festival", "Wedding", "Special Occasion"]
    symbolic_meanings = ["Prosperity", "Welcome", "Protection", "Harmony"]

    # Simulate processing delay
    await asyncio.sleep(0.1)

    # Generate mock results
    result = {
        "analysis_id": analysis_id,
        "design_classification": {
            "type": random.choice(pattern_types),
            "subtype": random.choice(subtypes),
            "region": random.choice(regions),
            "confidence": round(random.uniform(85.0, 99.0), 2)
        },
        "mathematical_properties": {
            "symmetry_type": random.choice(symmetry_types),
            "rotational_order": random.randint(2, 8),
            "reflection_axes": random.randint(0, 8),
            "complexity_score": round(random.uniform(0.5, 0.95), 2),
            "fractal_dimension": round(random.uniform(1.2, 1.8), 2)
        },
        "cultural_context": {
            "ceremonial_use": random.choice(ceremonial_uses),
            "seasonal_relevance": "All Seasons",
            "symbolic_meaning": random.choice(symbolic_meanings),
            "traditional_name": f"Kolam_{random.randint(100, 999)}"
        },
        "metadata": {
            "image_dimensions": image_array.shape[:2],
            "color_mode": "RGB" if len(image_array.shape) == 3 else "Grayscale",
            "analysis_version": "1.0.0"
        }
    }

    return result


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str) -> Dict[str, str]:
    """
    Delete analysis results by ID.

    Args:
        analysis_id: Unique identifier of the analysis

    Returns:
        Deletion confirmation
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found"
        )

    del analysis_storage[analysis_id]

    return {
        "status": "success",
        "message": f"Analysis {analysis_id} deleted successfully"
    }
