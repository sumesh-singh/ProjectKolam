"""
Main API router that combines all endpoint routers
"""
from fastapi import APIRouter
from app.api.v1.endpoints import patterns

api_router = APIRouter()

# Include pattern analysis endpoints
api_router.include_router(
    patterns.router,
    prefix="/patterns",
    tags=["patterns"]
)

# TODO: Add other endpoint routers when they are implemented
# api_router.include_router(
#     designs.router,
#     prefix="/designs",
#     tags=["designs"]
# )
# api_router.include_router(
#     health.router,
#     prefix="/health",
#     tags=["health"]
# )
