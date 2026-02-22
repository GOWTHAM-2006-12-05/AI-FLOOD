"""
FastAPI application entry point.

Run with:
    uvicorn backend.app.main:app --reload --port 8000

Or from the project root:
    python -m uvicorn backend.app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Core infrastructure ──
from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging, get_logger
from backend.app.core.errors import register_error_handlers
from backend.app.core.middleware import RequestLoggingMiddleware
from backend.app.core.health import run_health_check

# ── API routers ──
from backend.app.api.v1.disasters import router as disaster_router
from backend.app.api.v1.weather import router as weather_router
from backend.app.api.v1.flood import router as flood_router
from backend.app.api.v1.grid import router as grid_router
from backend.app.api.v1.forecast import router as forecast_router
from backend.app.api.v1.earthquake import router as earthquake_router
from backend.app.api.v1.cyclone import router as cyclone_router
from backend.app.api.v1.risk import router as risk_router
from backend.app.api.v1.alerts import router as alert_router
from backend.app.api.v1.assess import router as assess_router  # Full pipeline endpoint

# ── Initialise logging ──
setup_logging()
logger = get_logger(__name__)


# ── Application lifespan (startup / shutdown) ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    logger.info(
        "Starting %s v%s [%s]",
        settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT,
    )
    # Startup: initialise DB, Redis, load models here
    yield
    # Shutdown: close connections
    logger.info("Shutting down %s", settings.APP_NAME)


# ── Create application ──

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Hyper-local multi-disaster early warning system. "
        "Provides location-based radius filtering, real-time weather "
        "ingestion from Open-Meteo, ML feature engineering, "
        "XGBoost + LSTM ensemble flood prediction, "
        "1 km × 1 km hyper-local grid-based flood simulation, "
        "sub-hour multi-horizon flood forecasting, "
        "USGS earthquake monitoring with impact estimation, "
        "cyclone detection with IMD-scale escalation, "
        "unified risk aggregation, and "
        "multi-channel alert broadcasting."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware stack (order matters — outermost first) ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if not settings.CORS_ALLOW_ALL else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# ── Error handlers ──
register_error_handlers(app)

# ── Register routers ──
app.include_router(disaster_router)
app.include_router(weather_router)
app.include_router(flood_router)
app.include_router(grid_router)
app.include_router(forecast_router)
app.include_router(earthquake_router)
app.include_router(cyclone_router)
app.include_router(risk_router)
app.include_router(alert_router)
app.include_router(assess_router)  # Full pipeline risk assessment


# ── Root & health endpoints ──

@app.get("/", tags=["root"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "modules": [
            "radius-filter",
            "weather-ingestion",
            "flood-prediction",
            "grid-simulation",
            "sub-hour-forecast",
            "earthquake-monitoring",
            "cyclone-monitoring",
            "risk-aggregation",
            "alert-broadcasting",
        ],
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Deep health probe — checks all subsystems."""
    report = await run_health_check()
    return report.to_dict()


@app.get("/health/live", tags=["health"])
async def liveness():
    """Kubernetes liveness probe — is the process alive?"""
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
async def readiness():
    """Kubernetes readiness probe — can we serve traffic?"""
    report = await run_health_check()
    if report.status.value == "unhealthy":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=report.to_dict())
    return report.to_dict()
