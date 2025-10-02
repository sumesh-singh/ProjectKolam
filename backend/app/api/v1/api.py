"""
Main API router that combines all endpoint routers
"""
from fastapi import APIRouter
from app.api.v1.endpoints import patterns, designs, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    patterns.router,
    prefix="/patterns",
    tags=["patterns"]
)
api_router.include_router(
    designs.router,
    prefix="/designs",
    tags=["designs"]
)
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)
