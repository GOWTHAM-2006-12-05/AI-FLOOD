"""
Environment configuration — single source of truth for all settings.

Uses pydantic-settings for type-safe config with .env file support.
All values have sensible defaults for local development.

Usage:
    from backend.app.core.config import settings
    print(settings.DATABASE_URL)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables or .env file.

    Precedence: env var > .env file > default value
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ──
    APP_NAME: str = "AI Disaster Prediction"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"  # development | staging | production
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
    SECRET_KEY: str = "dev-secret-key-change-in-production"

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # uvicorn workers (production: 4+)
    RELOAD: bool = True  # auto-reload on file changes (dev only)

    # ── CORS ──
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]
    CORS_ALLOW_ALL: bool = True  # False in production

    # ── Database ──
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/disaster_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False  # log SQL queries

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 300  # default cache TTL in seconds (5 min)
    REDIS_WEATHER_TTL: int = 600  # weather data cache (10 min)
    REDIS_PREDICTION_TTL: int = 120  # ML prediction cache (2 min)

    # ── External APIs ──
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"
    USGS_EARTHQUAKE_URL: str = "https://earthquake.usgs.gov/fdsnws/event/1"
    WEATHER_FETCH_TIMEOUT: int = 30  # seconds

    # ── ML Models ──
    MODEL_DIR: str = "models"
    XGB_MODEL_PATH: str = "models/xgb_flood.joblib"
    LSTM_MODEL_PATH: str = "models/lstm_flood.joblib"
    SCALER_PATH: str = "models/scaler.joblib"
    ENSEMBLE_ALPHA: float = 0.65  # XGBoost weight in ensemble

    # ── Alert Broadcasting ──
    SMS_PROVIDER: str = "simulation"  # simulation | twilio | msg91
    SMS_API_KEY: Optional[str] = None
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SIREN_API_URL: str = "http://localhost:9090/api/sirens"
    VAPID_PRIVATE_KEY: Optional[str] = None
    VAPID_PUBLIC_KEY: Optional[str] = None

    # ── Spatial ──
    DEFAULT_RADIUS_KM: float = 50.0
    GRID_RESOLUTION_KM: float = 1.0  # hyper-local grid cell size

    # ── Monitoring ──
    ENABLE_METRICS: bool = True
    METRICS_PREFIX: str = "disaster_api"
    SENTRY_DSN: Optional[str] = None

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
