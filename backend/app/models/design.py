"""
Design database models
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class Design(Base):
    """
    Design model for storing Kolam design metadata
    """
    __tablename__ = "designs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    # kolam, muggu, rangoli, rangavalli
    type = Column(String(50), nullable=False, index=True)
    region = Column(String(100), index=True)
    complexity_level = Column(Integer, nullable=False)  # 1-5
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    status = Column(String(20), default="draft")  # draft, published, archived
    tags = Column(ARRAY(String), default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)

    # Relationships
    creator = relationship("User", back_populates="designs")
    mathematical_properties = relationship(
        "MathematicalProperty", back_populates="design", uselist=False)
    cultural_information = relationship(
        "CulturalInformation", back_populates="design", uselist=False)
    pattern_analyses = relationship("PatternAnalysis", back_populates="design")
    user_activities = relationship("UserActivity", back_populates="design")
    expert_validations = relationship(
        "ExpertValidation", back_populates="design")

    def __repr__(self):
        return f"<Design(id={self.id}, name={self.name}, type={self.type})>"


class MathematicalProperty(Base):
    """
    Mathematical properties of designs
    """
    __tablename__ = "mathematical_properties"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey(
        "designs.id"), nullable=False, unique=True)
    symmetry_type = Column(String(50))
    rotational_order = Column(Integer)
    reflection_axes = Column(Integer)
    grid_dimensions = Column(String)  # JSON string for (width, height)
    complexity_score = Column(Float)
    geometric_features = Column(Text)  # JSON string
    mathematical_descriptors = Column(Text)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    design = relationship("Design", back_populates="mathematical_properties")

    def __repr__(self):
        return f"<MathematicalProperty(design_id={self.design_id}, symmetry_type={self.symmetry_type})>"


class CulturalInformation(Base):
    """
    Cultural information and metadata for designs
    """
    __tablename__ = "cultural_information"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey(
        "designs.id"), nullable=False, unique=True)
    origin_region = Column(String(100))
    historical_period = Column(String(100))
    ceremonial_use = Column(Text)
    symbolic_meaning = Column(Text)
    traditional_stories = Column(Text)  # JSON string
    regional_variations = Column(Text)  # JSON string
    expert_validated = Column(Boolean, default=False)
    validation_date = Column(DateTime(timezone=True))
    validated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    design = relationship("Design", back_populates="cultural_information")
    validator = relationship("User")

    def __repr__(self):
        return f"<CulturalInformation(design_id={self.design_id}, origin_region={self.origin_region})>"


class PatternAnalysis(Base):
    """
    Results from pattern analysis operations
    """
    __tablename__ = "pattern_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey(
        "designs.id"), nullable=False)
    analysis_type = Column(String(50), nullable=False)
    confidence_score = Column(Float)
    analysis_data = Column(Text, nullable=False)  # JSON string
    processing_time_ms = Column(Integer)
    model_version = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    design = relationship("Design", back_populates="pattern_analyses")

    def __repr__(self):
        return f"<PatternAnalysis(design_id={self.design_id}, type={self.analysis_type})>"


class UserActivity(Base):
    """
    User activities and social features
    """
    __tablename__ = "user_activities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey(
        "users.id"), nullable=False)
    design_id = Column(UUID(as_uuid=True), ForeignKey(
        "designs.id"), nullable=False)
    # view, like, share, comment
    activity_type = Column(String(20), nullable=False)
    activity_data = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="activities")
    design = relationship("Design", back_populates="user_activities")

    def __repr__(self):
        return f"<UserActivity(user_id={self.user_id}, design_id={self.design_id}, type={self.activity_type})>"


class ExpertValidation(Base):
    """
    Expert validations for cultural accuracy
    """
    __tablename__ = "expert_validations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey(
        "designs.id"), nullable=False)
    expert_id = Column(UUID(as_uuid=True), ForeignKey(
        "users.id"), nullable=False)
    # pending, approved, rejected
    validation_status = Column(String(20), nullable=False)
    feedback = Column(Text)
    cultural_accuracy_score = Column(Integer)  # 1-5
    authenticity_score = Column(Integer)  # 1-5
    recommendations = Column(Text)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())

    # Relationships
    design = relationship("Design", back_populates="expert_validations")
    expert = relationship("User")

    def __repr__(self):
        return f"<ExpertValidation(design_id={self.design_id}, expert_id={self.expert_id}, status={self.validation_status})>"
