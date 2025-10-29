"""
Production Configuration Management
Handles environment-based configuration with validation
"""

import os
from typing import Optional
from pydantic import BaseSettings, validator
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Configuration
    api_title: str = "Autonomous Agent Platform"
    api_version: str = "2.0.0"
    api_description: str = "Enterprise-grade autonomous agent platform"
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    api_workers: int = int(os.getenv("API_WORKERS", 4))
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = environment == "development"
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./agent.db")
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", 20))
    database_max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", 40))
    
    # Redis Cache
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"
    cache_ttl: int = int(os.getenv("CACHE_TTL", 3600))
    
    # AI Models
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    model_name: str = os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
    max_tokens: int = int(os.getenv("MAX_TOKENS", 2000))
    
    # Authentication
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", 24))
    
    # Security
    cors_origins: list = os.getenv("CORS_ORIGINS", "*").split(",")
    allowed_hosts: list = os.getenv("ALLOWED_HOSTS", "*").split(",")
    
    # Rate Limiting
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    rate_limit_period: int = int(os.getenv("RATE_LIMIT_PERIOD", 60))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "json"  # json or text
    
    # Monitoring
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    
    # Task Configuration
    max_task_timeout: int = int(os.getenv("MAX_TASK_TIMEOUT", 300))
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 100))
    task_queue_size: int = int(os.getenv("TASK_QUEUE_SIZE", 1000))
    
    # Feature Flags
    enable_browser_automation: bool = os.getenv("ENABLE_BROWSER_AUTOMATION", "true").lower() == "true"
    enable_code_execution: bool = os.getenv("ENABLE_CODE_EXECUTION", "true").lower() == "true"
    enable_data_analysis: bool = os.getenv("ENABLE_DATA_ANALYSIS", "true").lower() == "true"
    
    @validator("anthropic_api_key")
    def validate_api_key(cls, v):
        if not v and cls.environment == "production":
            raise ValueError("ANTHROPIC_API_KEY is required in production")
        return v
    
    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v):
        if v == "your-secret-key-change-in-production":
            logger.warning("Using default JWT secret key - change in production!")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Log configuration on startup
if settings.environment == "production":
    logger.info(f"Running in PRODUCTION mode")
    logger.info(f"Database: {settings.database_url[:50]}...")
    logger.info(f"Redis: {'enabled' if settings.redis_enabled else 'disabled'}")
else:
    logger.info(f"Running in DEVELOPMENT mode")
    logger.debug(f"Debug mode: {settings.debug}")
