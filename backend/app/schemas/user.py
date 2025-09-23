"""
Pydantic schemas for User API
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID


class UserBase(BaseModel):
    """
    Base user schema
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    role: str = Field(default="user", pattern="^(user|expert|admin)$")
    is_active: bool = True


class UserCreate(UserBase):
    """
    Schema for creating a new user
    """
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """
    Schema for updating user information
    """
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    role: Optional[str] = Field(None, pattern="^(user|expert|admin)$")
    is_active: Optional[bool] = None
    profile_data: Optional[dict] = None


class UserInDBBase(UserBase):
    """
    Base schema for user in database
    """
    id: UUID
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class User(UserInDBBase):
    """
    User schema for API responses
    """
    pass


class UserInDB(UserInDBBase):
    """
    User schema with password hash (for internal use)
    """
    password_hash: str


class Token(BaseModel):
    """
    JWT token schema
    """
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """
    JWT token payload
    """
    username: Optional[str] = None
    user_id: Optional[UUID] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    """
    Login request schema
    """
    username: str
    password: str


class PasswordResetRequest(BaseModel):
    """
    Password reset request schema
    """
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """
    Password reset confirmation schema
    """
    token: str
    new_password: str = Field(..., min_length=8)
