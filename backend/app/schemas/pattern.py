"""
Pydantic schemas for Pattern Analysis API
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class DesignClassification(BaseModel):
    """
    Design classification results
    """
    type: str
    subtype: str
    region: str
    confidence: float


class MathematicalProperties(BaseModel):
    """
    Mathematical properties of the pattern
    """
    symmetry_type: str
    rotational_order: int
    reflection_axes: int
    complexity_score: float
    fractal_dimension: Optional[float] = None


class CulturalContext(BaseModel):
    """
    Cultural context information
    """
    ceremonial_use: str
    seasonal_relevance: str
    symbolic_meaning: str
    traditional_name: Optional[str] = None


class AnalysisResult(BaseModel):
    """
    Complete pattern analysis result
    """
    analysis_id: str
    design_classification: DesignClassification
    mathematical_properties: MathematicalProperties
    cultural_context: CulturalContext
    metadata: Optional[Dict[str, Any]] = None


class PatternAnalysisRequest(BaseModel):
    """
    Request schema for pattern analysis
    """
    image_data: Optional[str] = None
    analysis_options: Optional[Dict[str, Any]] = None


class PatternAnalysisResponse(BaseModel):
    """
    Response schema for pattern analysis
    """
    status: str
    message: str
    analysis_id: str
    processing_time_ms: int
    result: AnalysisResult
    timestamp: datetime


class BatchAnalysisResult(BaseModel):
    """
    Result for individual item in batch analysis
    """
    filename: str
    status: str
    analysis_id: Optional[str] = None
    error: Optional[str] = None


class BatchAnalysisResponse(BaseModel):
    """
    Response schema for batch pattern analysis
    """
    status: str
    total: int
    successful: int
    failed: int
    results: List[BatchAnalysisResult]
    timestamp: datetime
