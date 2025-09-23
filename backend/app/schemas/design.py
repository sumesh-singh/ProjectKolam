"""
Pydantic schemas for Design API
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID


class DesignBase(BaseModel):
    """
    Base design schema
    """
    name: str = Field(..., max_length=255)
    type: str = Field(..., pattern="^(kolam|muggu|rangoli|rangavalli)$")
    region: Optional[str] = Field(None, max_length=100)
    complexity_level: int = Field(..., ge=1, le=5)
    tags: List[str] = Field(default_factory=list)


class DesignCreate(DesignBase):
    """
    Schema for creating a new design
    """
    pass


class DesignUpdate(BaseModel):
    """
    Schema for updating design information
    """
    name: Optional[str] = Field(None, max_length=255)
    type: Optional[str] = Field(
        None, pattern="^(kolam|muggu|rangoli|rangavalli)$")
    region: Optional[str] = Field(None, max_length=100)
    complexity_level: Optional[int] = Field(None, ge=1, le=5)
    status: Optional[str] = Field(None, pattern="^(draft|published|archived)$")
    tags: Optional[List[str]] = None


class DesignInDBBase(DesignBase):
    """
    Base schema for design in database
    """
    id: UUID
    creator_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime
    view_count: int
    like_count: int

    class Config:
        from_attributes = True


class Design(DesignInDBBase):
    """
    Design schema for API responses
    """
    pass


class DesignWithDetails(Design):
    """
    Design schema with related data
    """
    mathematical_properties: Optional[dict] = None
    cultural_information: Optional[dict] = None
    creator_username: Optional[str] = None


class MathematicalPropertyBase(BaseModel):
    """
    Base mathematical properties schema
    """
    symmetry_type: Optional[str] = None
    rotational_order: Optional[int] = None
    reflection_axes: Optional[int] = None
    grid_dimensions: Optional[str] = None
    complexity_score: Optional[float] = None
    geometric_features: Optional[dict] = None
    mathematical_descriptors: Optional[dict] = None


class MathematicalPropertyCreate(MathematicalPropertyBase):
    """
    Schema for creating mathematical properties
    """
    design_id: UUID


class MathematicalProperty(MathematicalPropertyBase):
    """
    Mathematical properties schema
    """
    id: UUID
    design_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class CulturalInformationBase(BaseModel):
    """
    Base cultural information schema
    """
    origin_region: Optional[str] = None
    historical_period: Optional[str] = None
    ceremonial_use: Optional[str] = None
    symbolic_meaning: Optional[str] = None
    traditional_stories: Optional[dict] = None
    regional_variations: Optional[dict] = None


class CulturalInformationCreate(CulturalInformationBase):
    """
    Schema for creating cultural information
    """
    design_id: UUID


class CulturalInformation(CulturalInformationBase):
    """
    Cultural information schema
    """
    id: UUID
    design_id: UUID
    expert_validated: bool
    validation_date: Optional[datetime] = None
    validated_by: Optional[UUID] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PatternAnalysisBase(BaseModel):
    """
    Base pattern analysis schema
    """
    analysis_type: str
    confidence_score: Optional[float] = None
    analysis_data: dict
    processing_time_ms: Optional[int] = None
    model_version: Optional[str] = None


class PatternAnalysisCreate(PatternAnalysisBase):
    """
    Schema for creating pattern analysis
    """
    design_id: UUID


class PatternAnalysis(PatternAnalysisBase):
    """
    Pattern analysis schema
    """
    id: UUID
    design_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class DesignSearchFilters(BaseModel):
    """
    Filters for design search
    """
    type: Optional[str] = None
    region: Optional[str] = None
    complexity: Optional[int] = None
    creator_id: Optional[UUID] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None


class DesignSearchResponse(BaseModel):
    """
    Response for design search
    """
    designs: List[DesignWithDetails]
    total_count: int
    page: int
    limit: int
