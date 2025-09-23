"""
Main API router
"""
from fastapi import APIRouter

from app.api.v1.endpoints import users, patterns

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(
    patterns.router, prefix="/patterns", tags=["patterns"])
