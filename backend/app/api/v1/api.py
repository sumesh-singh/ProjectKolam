"""
Main API router
"""
from fastapi import APIRouter

# Import only the modules that exist and don't have database dependencies
from app.api.v1.endpoints import patterns

api_router = APIRouter()

# Include only the working endpoint routers
api_router.include_router(
    patterns.router, prefix="/patterns", tags=["patterns"])

# Note: kolam_ai and users endpoints are disabled due to missing dependencies
# They can be re-enabled once the required models and database are properly configured
