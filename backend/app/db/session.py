"""
Database session configuration and connection management
"""
from typing import AsyncGenerator
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# PostgreSQL async engine
engine = create_async_engine(
    settings.sql_database_url.replace(
        "postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG,
    future=True,
)

# Async session factory
async_session_factory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# MongoDB client
mongodb_client: AsyncIOMotorClient = AsyncIOMotorClient(settings.MONGODB_URL)
mongodb_database = mongodb_client[settings.MONGODB_DATABASE]

# Redis client
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database session
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_mongodb() -> AsyncIOMotorClient:
    """
    Dependency for getting MongoDB client
    """
    return mongodb_database


async def get_redis() -> redis.Redis:
    """
    Dependency for getting Redis client
    """
    return redis_client


async def close_database_connections():
    """
    Close all database connections (for cleanup)
    """
    await engine.dispose()
    mongodb_client.close()
    await redis_client.aclose()
