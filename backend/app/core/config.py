"""
Core configuration settings for the Kolam Design System
"""
from typing import List, Optional, Union
from pydantic import AnyHttpUrl, field_validator, ValidationInfo
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """

    # Project
    PROJECT_NAME: str = "Kolam Design Pattern Recognition and Recreation System"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    DEBUG: bool = True

    # Server
    SERVER_NAME: str = "localhost"
    SERVER_HOST: AnyHttpUrl = "http://localhost"
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Vite dev server
        "http://localhost:8000",  # FastAPI
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(
        cls, v: Union[str, List[str]]
    ) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database - PostgreSQL
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "kolam_user"
    POSTGRES_PASSWORD: str = "kolam_password"
    POSTGRES_DB: str = "kolam_db"
    POSTGRES_PORT: str = "5432"
    DATABASE_URL: Optional[str] = None

    @property
    def sql_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Database - MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "kolam_db"

    # Cache - Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # Authentication
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days

    # OAuth
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None

    # File Storage
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_DEFAULT_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "kolam-designs"
    USE_LOCAL_STORAGE: bool = True
    LOCAL_STORAGE_PATH: str = "./uploads"

    # ML Model Configuration
    MODEL_CACHE_DIR: str = "./models"
    PATTERN_CLASSIFICATION_MODEL: str = "pattern_classifier_v1"
    SYMMETRY_DETECTION_MODEL: str = "symmetry_detector_v1"
    MAX_IMAGE_SIZE: int = 4096
    SUPPORTED_IMAGE_FORMATS: List[str] = ["JPEG", "PNG", "TIFF", "BMP"]

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Email (for notifications)
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
