"""
Calendar-based Weather & Flood Prediction System.

This module provides:
- Historical weather data ingestion from Open-Meteo Archive API
- Future forecast fetching from Open-Meteo Forecast API
- Flood probability calculation per date
- Calendar API endpoints for frontend
- Background jobs for scheduled data refresh
"""

from .models import HistoricalWeather, WeatherForecast, CalendarDayData, CalendarMonthData, RiskLevel
from .historical_service import HistoricalWeatherService
from .forecast_service import ForecastService
from .flood_calculator import FloodRiskCalculator
from .background_jobs import BackgroundJobManager, get_job_manager, ScheduledJobRunner

__all__ = [
    "HistoricalWeather",
    "WeatherForecast",
    "CalendarDayData",
    "CalendarMonthData",
    "RiskLevel",
    "HistoricalWeatherService",
    "ForecastService",
    "FloodRiskCalculator",
    "BackgroundJobManager",
    "get_job_manager",
    "ScheduledJobRunner",
]
