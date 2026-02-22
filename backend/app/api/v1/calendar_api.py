"""
Calendar API Endpoints.

═══════════════════════════════════════════════════════════════════════════
API DESIGN
═══════════════════════════════════════════════════════════════════════════

Endpoints:

1. GET /api/v1/calendar/month?year=2026&month=2&lat=12.9&lon=80.2
   Returns full month of data (historical + forecast)

2. GET /api/v1/calendar/day?date=2026-02-22&lat=12.9&lon=80.2
   Returns single day data (historical or forecast)

3. GET /api/v1/calendar/forecast?lat=12.9&lon=80.2
   Returns next 5 days forecast

4. GET /api/v1/calendar/historical?lat=12.9&lon=80.2&year=2025
   Returns full year historical data

5. POST /api/v1/calendar/ingest
   Triggers background ingestion of historical data

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from backend.app.calendar.models import CalendarDayData, CalendarMonthData, RiskLevel
from backend.app.calendar.forecast_service import ForecastService
from backend.app.calendar.historical_service import HistoricalWeatherService
from backend.app.calendar.flood_calculator import FloodRiskCalculator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/calendar", tags=["calendar"])

# ═══════════════════════════════════════════════════════════════════════════
# Shared Services (singleton pattern)
# ═══════════════════════════════════════════════════════════════════════════

_forecast_service: Optional[ForecastService] = None
_historical_service: Optional[HistoricalWeatherService] = None
_flood_calculator: Optional[FloodRiskCalculator] = None


def get_forecast_service() -> ForecastService:
    global _forecast_service
    if _forecast_service is None:
        _forecast_service = ForecastService()
    return _forecast_service


def get_historical_service() -> HistoricalWeatherService:
    global _historical_service
    if _historical_service is None:
        _historical_service = HistoricalWeatherService()
    return _historical_service


def get_flood_calculator() -> FloodRiskCalculator:
    global _flood_calculator
    if _flood_calculator is None:
        _flood_calculator = FloodRiskCalculator(use_ml=True)
    return _flood_calculator


# ═══════════════════════════════════════════════════════════════════════════
# Response Schemas
# ═══════════════════════════════════════════════════════════════════════════

class DayDataResponse(BaseModel):
    """Response for single day weather/flood data."""
    date: str
    is_forecast: bool
    is_today: bool
    rain_mm: float
    temperature_max: float
    temperature_min: float
    flood_probability: float
    risk_level: str
    risk_color: str
    weather_description: str = ""
    has_data: bool = True
    contributing_factors: List[Dict[str, Any]] = []


class MonthDataResponse(BaseModel):
    """Response for full month calendar data."""
    year: int
    month: int
    latitude: float
    longitude: float
    days: List[DayDataResponse]
    summary: Dict[str, Any]


class ForecastResponse(BaseModel):
    """Response for 5-day forecast."""
    latitude: float
    longitude: float
    forecasts: List[DayDataResponse]
    fetched_at: str
    summary: Dict[str, Any]


class IngestionRequest(BaseModel):
    """Request to ingest historical data."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    start_year: int = Field(2020, ge=1940, le=2100)
    end_year: Optional[int] = Field(None, ge=1940, le=2100)


class IngestionStatusResponse(BaseModel):
    """Response for ingestion status."""
    status: str
    message: str
    progress: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/month",
    response_model=MonthDataResponse,
    summary="Get Month Calendar Data",
    description="Get weather and flood risk data for a full month. Returns historical data for past dates and forecasts for future dates.",
)
async def get_month_data(
    year: int = Query(..., ge=1940, le=2100, description="Year"),
    month: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
):
    """
    Get full month of calendar data.
    
    Logic:
    - For past dates: Query historical database or fetch from archive API
    - For today: Use current weather + forecast
    - For future dates (up to 5 days): Use forecast API
    - For dates > 5 days in future: Return no data
    """
    today = date.today()
    
    # Determine days in month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    
    month_start = date(year, month, 1)
    days_in_month = (next_month - month_start).days
    
    days_data: List[DayDataResponse] = []
    
    forecast_service = get_forecast_service()
    historical_service = get_historical_service()
    
    # Pre-fetch forecast if month contains future dates
    forecast_data = {}
    if month_start <= today + timedelta(days=5):
        try:
            forecast_result = await forecast_service.get_forecast(lat, lon)
            for f in forecast_result.forecasts:
                forecast_data[f.date] = f
        except Exception as e:
            logger.warning("Failed to fetch forecast: %s", e)
    
    # Pre-fetch historical data for the month
    historical_data = {}
    if month_start.year <= today.year:
        try:
            historical_records = await historical_service.get_month_data(
                latitude=lat,
                longitude=lon,
                year=year,
                month=month,
            )
            for h in historical_records:
                historical_data[h.date] = h
        except Exception as e:
            logger.warning("Failed to fetch historical data: %s", e)
    
    # Build response for each day
    for day in range(1, days_in_month + 1):
        current_date = date(year, month, day)
        is_today = current_date == today
        is_future = current_date > today
        
        day_response = DayDataResponse(
            date=current_date.isoformat(),
            is_forecast=is_future,
            is_today=is_today,
            rain_mm=0.0,
            temperature_max=0.0,
            temperature_min=0.0,
            flood_probability=0.0,
            risk_level="safe",
            risk_color="#6b7280",
            weather_description="",
            has_data=False,
        )
        
        if is_future:
            # Use forecast data
            if current_date in forecast_data:
                f = forecast_data[current_date]
                day_response = DayDataResponse(
                    date=current_date.isoformat(),
                    is_forecast=True,
                    is_today=False,
                    rain_mm=round(f.rain_sum, 2),
                    temperature_max=round(f.temperature_max, 1),
                    temperature_min=round(f.temperature_min, 1),
                    flood_probability=round(f.flood_probability * 100, 1),
                    risk_level=f.risk_level.value,
                    risk_color=f.risk_color,
                    weather_description=f.weather_description,
                    has_data=True,
                )
        else:
            # Use historical data
            if current_date in historical_data:
                h = historical_data[current_date]
                day_response = DayDataResponse(
                    date=current_date.isoformat(),
                    is_forecast=False,
                    is_today=is_today,
                    rain_mm=round(h.rain_sum, 2),
                    temperature_max=round(h.temperature_max, 1),
                    temperature_min=round(h.temperature_min, 1),
                    flood_probability=round(h.flood_probability * 100, 1),
                    risk_level=h.risk_level.value,
                    risk_color=h.risk_color,
                    weather_description="",
                    has_data=True,
                )
        
        days_data.append(day_response)
    
    # Calculate summary
    days_with_data = [d for d in days_data if d.has_data]
    summary = {
        "total_rain_mm": round(sum(d.rain_mm for d in days_with_data), 2),
        "avg_temp_max": round(
            sum(d.temperature_max for d in days_with_data) / max(1, len(days_with_data)), 1
        ),
        "high_risk_days": len([d for d in days_with_data if d.flood_probability >= 35]),
        "severe_days": len([d for d in days_with_data if d.flood_probability >= 60]),
        "days_with_data": len(days_with_data),
        "days_without_data": len(days_data) - len(days_with_data),
    }
    
    return MonthDataResponse(
        year=year,
        month=month,
        latitude=round(lat, 6),
        longitude=round(lon, 6),
        days=days_data,
        summary=summary,
    )


@router.get(
    "/day",
    response_model=DayDataResponse,
    summary="Get Single Day Data",
    description="Get weather and flood risk data for a specific date.",
)
async def get_day_data(
    date_str: str = Query(
        ...,
        alias="date",
        description="Date in ISO format (YYYY-MM-DD)",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
):
    """
    Get weather/flood data for a single date.
    
    Logic:
    - If date > today: Fetch from forecast API
    - If date <= today: Query historical database or archive API
    """
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    today = date.today()
    is_today = target_date == today
    is_future = target_date > today
    
    if is_future:
        # Use forecast
        forecast_service = get_forecast_service()
        
        days_ahead = (target_date - today).days
        if days_ahead > 7:
            return DayDataResponse(
                date=target_date.isoformat(),
                is_forecast=True,
                is_today=False,
                rain_mm=0.0,
                temperature_max=0.0,
                temperature_min=0.0,
                flood_probability=0.0,
                risk_level="safe",
                risk_color="#6b7280",
                weather_description="",
                has_data=False,
            )
        
        try:
            forecast = await forecast_service.get_forecast_day(lat, lon, target_date)
            
            if forecast:
                # Get detailed risk factors
                calculator = get_flood_calculator()
                from .flood_calculator import FloodRiskInput
                
                inputs = FloodRiskInput(
                    rain_sum=forecast.rain_sum,
                    temperature_max=forecast.temperature_max,
                    temperature_min=forecast.temperature_min,
                    humidity_mean=60.0,  # Estimate
                    pressure_mean=1013.25,  # Estimate
                    wind_speed_max=forecast.wind_speed_max,
                )
                risk_result = calculator.calculate(inputs)
                
                return DayDataResponse(
                    date=target_date.isoformat(),
                    is_forecast=True,
                    is_today=False,
                    rain_mm=round(forecast.rain_sum, 2),
                    temperature_max=round(forecast.temperature_max, 1),
                    temperature_min=round(forecast.temperature_min, 1),
                    flood_probability=round(forecast.flood_probability * 100, 1),
                    risk_level=forecast.risk_level.value,
                    risk_color=forecast.risk_color,
                    weather_description=forecast.weather_description,
                    has_data=True,
                    contributing_factors=risk_result.contributing_factors,
                )
            else:
                raise HTTPException(status_code=404, detail="No forecast data available")
                
        except Exception as e:
            logger.error("Failed to get forecast for %s: %s", target_date, e)
            raise HTTPException(status_code=500, detail=str(e))
    
    else:
        # Use historical
        historical_service = get_historical_service()
        
        try:
            record = await historical_service.get_historical_day(lat, lon, target_date)
            
            if record:
                # Get detailed risk factors
                calculator = get_flood_calculator()
                from .flood_calculator import FloodRiskInput
                
                inputs = FloodRiskInput(
                    rain_sum=record.rain_sum,
                    temperature_max=record.temperature_max,
                    temperature_min=record.temperature_min,
                    humidity_mean=record.humidity_mean,
                    pressure_mean=record.pressure_mean,
                    wind_speed_max=record.wind_speed_max,
                )
                risk_result = calculator.calculate(inputs)
                
                return DayDataResponse(
                    date=target_date.isoformat(),
                    is_forecast=False,
                    is_today=is_today,
                    rain_mm=round(record.rain_sum, 2),
                    temperature_max=round(record.temperature_max, 1),
                    temperature_min=round(record.temperature_min, 1),
                    flood_probability=round(record.flood_probability * 100, 1),
                    risk_level=record.risk_level.value,
                    risk_color=record.risk_color,
                    weather_description="",
                    has_data=True,
                    contributing_factors=risk_result.contributing_factors,
                )
            else:
                return DayDataResponse(
                    date=target_date.isoformat(),
                    is_forecast=False,
                    is_today=is_today,
                    rain_mm=0.0,
                    temperature_max=0.0,
                    temperature_min=0.0,
                    flood_probability=0.0,
                    risk_level="safe",
                    risk_color="#6b7280",
                    weather_description="",
                    has_data=False,
                )
                
        except Exception as e:
            logger.error("Failed to get historical data for %s: %s", target_date, e)
            raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/forecast",
    response_model=ForecastResponse,
    summary="Get 5-Day Forecast",
    description="Get weather forecast and flood risk for the next 5 days.",
)
async def get_forecast(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    refresh: bool = Query(False, description="Force refresh (bypass cache)"),
):
    """
    Get 5-day forecast with flood risk assessment.
    """
    forecast_service = get_forecast_service()
    
    try:
        result = await forecast_service.get_forecast(lat, lon, force_refresh=refresh)
        
        forecasts = []
        for f in result.forecasts:
            forecasts.append(DayDataResponse(
                date=f.date.isoformat(),
                is_forecast=True,
                is_today=f.date == date.today(),
                rain_mm=round(f.rain_sum, 2),
                temperature_max=round(f.temperature_max, 1),
                temperature_min=round(f.temperature_min, 1),
                flood_probability=round(f.flood_probability * 100, 1),
                risk_level=f.risk_level.value,
                risk_color=f.risk_color,
                weather_description=f.weather_description,
                has_data=True,
            ))
        
        summary = {
            "total_rain_mm": round(sum(f.rain_sum for f in result.forecasts), 2),
            "max_rain_day": max(
                (f for f in result.forecasts),
                key=lambda f: f.rain_sum
            ).date.isoformat() if result.forecasts else None,
            "high_risk_days": len([f for f in result.forecasts if f.flood_probability >= 0.35]),
            "severe_risk_days": len([f for f in result.forecasts if f.flood_probability >= 0.60]),
        }
        
        return ForecastResponse(
            latitude=round(result.latitude, 6),
            longitude=round(result.longitude, 6),
            forecasts=forecasts,
            fetched_at=result.fetched_at.isoformat(),
            summary=summary,
        )
        
    except Exception as e:
        logger.error("Failed to get forecast: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/historical",
    summary="Get Historical Year Data",
    description="Get full year of historical weather data.",
)
async def get_historical_year(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    year: int = Query(..., ge=1940, le=2100, description="Year"),
):
    """
    Get full year of historical weather data.
    """
    historical_service = get_historical_service()
    
    try:
        records = await historical_service.fetch_year(lat, lon, year)
        
        return {
            "year": year,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "record_count": len(records),
            "data": [r.to_dict() for r in records],
            "summary": {
                "total_rain_mm": round(sum(r.rain_sum for r in records), 2),
                "avg_temp_max": round(sum(r.temperature_max for r in records) / max(1, len(records)), 1),
                "high_risk_days": len([r for r in records if r.flood_probability >= 0.35]),
                "severe_days": len([r for r in records if r.flood_probability >= 0.60]),
            }
        }
        
    except Exception as e:
        logger.error("Failed to get historical data for year %d: %s", year, e)
        raise HTTPException(status_code=500, detail=str(e))


# Background ingestion tracking
_ingestion_status: Dict[str, Any] = {}


async def _run_ingestion(
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    task_id: str,
):
    """Background task for data ingestion."""
    global _ingestion_status
    
    _ingestion_status[task_id] = {
        "status": "running",
        "progress": 0,
        "current_year": start_year,
    }
    
    try:
        historical_service = get_historical_service()
        
        def on_progress(progress):
            _ingestion_status[task_id] = {
                "status": "running",
                "progress": progress.progress_pct,
                "current_year": progress.current_year,
                "completed": progress.completed_years,
                "total": progress.total_years,
            }
        
        result = await historical_service.ingest_years(
            latitude=latitude,
            longitude=longitude,
            start_year=start_year,
            end_year=end_year,
            on_progress=on_progress,
        )
        
        _ingestion_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "total_records": result.total_records,
            "completed_years": result.completed_years,
            "failed_years": result.failed_years,
        }
        
    except Exception as e:
        _ingestion_status[task_id] = {
            "status": "failed",
            "error": str(e),
        }


@router.post(
    "/ingest",
    response_model=IngestionStatusResponse,
    summary="Start Historical Data Ingestion",
    description="Start background ingestion of historical weather data.",
)
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start background ingestion of historical data.
    
    Returns a task ID to check progress.
    """
    import uuid
    
    task_id = str(uuid.uuid4())
    end_year = request.end_year or date.today().year
    
    background_tasks.add_task(
        _run_ingestion,
        request.latitude,
        request.longitude,
        request.start_year,
        end_year,
        task_id,
    )
    
    return IngestionStatusResponse(
        status="started",
        message=f"Ingestion started for years {request.start_year}-{end_year}",
        progress={"task_id": task_id},
    )


@router.get(
    "/ingest/status/{task_id}",
    response_model=IngestionStatusResponse,
    summary="Get Ingestion Status",
    description="Check the status of a background ingestion task.",
)
async def get_ingestion_status(task_id: str):
    """
    Get status of a background ingestion task.
    """
    if task_id not in _ingestion_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = _ingestion_status[task_id]
    
    return IngestionStatusResponse(
        status=status.get("status", "unknown"),
        message=f"Progress: {status.get('progress', 0):.1f}%",
        progress=status,
    )
