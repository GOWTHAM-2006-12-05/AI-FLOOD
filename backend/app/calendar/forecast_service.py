"""
Weather Forecast Service.

═══════════════════════════════════════════════════════════════════════════
OPEN-METEO FORECAST API
═══════════════════════════════════════════════════════════════════════════

Endpoint: https://api.open-meteo.com/v1/forecast

Features:
- 7-16 day forecast (depending on model)
- Hourly and daily resolution
- Multiple weather models available
- Free tier: 10,000 API calls/day
- No API key required

We use:
- Daily rain_sum for flood calculation
- Daily temp max/min for context
- Daily precipitation_probability for confidence
- Daily weather_code for icons
- Daily wind_speed_max for cyclone context

═══════════════════════════════════════════════════════════════════════════
FLOOD PROBABILITY FROM FORECAST
═══════════════════════════════════════════════════════════════════════════

How daily rainfall maps to flood probability:

1. DIRECT MAPPING (MVP)
   - Use forecast rain_sum like historical rainfall
   - Apply same threshold logic
   - Lower confidence due to forecast uncertainty

2. PROBABILISTIC ENHANCEMENT
   - Use precipitation_probability as confidence weight
   - Higher precip_prob + high rain = higher flood risk
   - Lower precip_prob = lower confidence regardless of rain amount

3. ML INTEGRATION
   - If ML models available, pass forecast features through ensemble
   - Features: rain_sum, temp, humidity, pressure, antecedent rain
   - Output: calibrated flood probability

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from .models import WeatherForecast, RiskLevel
from .flood_calculator import FloodRiskCalculator, FloodRiskInput

# Import Redis cache helpers
try:
    from backend.app.core.cache import cache_get, cache_set, cache_delete, cache_clear_prefix
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_FORECAST_PARAMS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
]

# How far ahead to forecast
MAX_FORECAST_DAYS = 7
RETURN_FORECAST_DAYS = 5  # Return only next 5 days to user

# Cache forecast for this many seconds
FORECAST_CACHE_TTL = 1800  # 30 minutes


@dataclass
class ForecastResult:
    """Result of a forecast fetch operation."""
    latitude: float
    longitude: float
    forecasts: List[WeatherForecast]
    fetched_at: datetime
    model: str  # Weather model used
    elevation: float
    
    @property
    def is_stale(self) -> bool:
        """Check if forecast is older than TTL."""
        age = (datetime.now() - self.fetched_at).total_seconds()
        return age > FORECAST_CACHE_TTL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "forecasts": [f.to_dict() for f in self.forecasts],
            "fetched_at": self.fetched_at.isoformat(),
            "model": self.model,
            "elevation": round(self.elevation, 1),
            "summary": {
                "total_rain_mm": round(sum(f.rain_sum for f in self.forecasts), 2),
                "max_rain_day": max((f for f in self.forecasts), key=lambda f: f.rain_sum).date.isoformat() if self.forecasts else None,
                "high_risk_days": len([f for f in self.forecasts if f.flood_probability >= 0.35]),
                "severe_risk_days": len([f for f in self.forecasts if f.flood_probability >= 0.60]),
            }
        }


class ForecastService:
    """
    Service for fetching weather forecasts and calculating flood risk.
    
    Usage:
        service = ForecastService()
        
        # Get 5-day forecast
        result = await service.get_forecast(lat=12.9, lon=80.2)
        for day in result.forecasts:
            print(f"{day.date}: Rain={day.rain_sum}mm, Flood={day.flood_probability*100:.0f}%")
    """
    
    def __init__(
        self,
        flood_calculator: Optional[FloodRiskCalculator] = None,
        cache_enabled: bool = True,
    ):
        self.flood_calculator = flood_calculator or FloodRiskCalculator(use_ml=True)
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, ForecastResult] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
    
    def _cache_key(self, lat: float, lon: float) -> str:
        """Generate cache key for location."""
        # Round to ~1km precision for cache hit
        return f"{lat:.2f},{lon:.2f}"
    
    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        force_refresh: bool = False,
    ) -> ForecastResult:
        """
        Get weather forecast for next 5 days.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            force_refresh: If True, bypass cache
            
        Returns:
            ForecastResult with 5-day forecast
        """
        # Check cache
        cache_key = self._cache_key(latitude, longitude)
        redis_key = f"forecast:{cache_key}"
        
        if self.cache_enabled and not force_refresh:
            # Try Redis first
            if REDIS_AVAILABLE:
                cached_dict = await cache_get(redis_key)
                if cached_dict:
                    logger.debug("Redis cache HIT for forecast %s", cache_key)
                    return self._dict_to_result(cached_dict)
            
            # Fall back to in-memory cache
            cached = self._cache.get(cache_key)
            if cached and not cached.is_stale:
                logger.debug("Memory cache HIT for forecast %s", cache_key)
                return cached
        
        # Build API request
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(DEFAULT_FORECAST_PARAMS),
            "timezone": "auto",
            "forecast_days": MAX_FORECAST_DAYS,
        }
        
        # Fetch from API
        client = await self._get_client()
        
        try:
            response = await client.get(FORECAST_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Forecast API error: %s", e)
            raise RuntimeError(f"Forecast API error: {e.response.status_code}")
        except Exception as e:
            logger.error("Forecast request failed: %s", e)
            raise
        
        # Parse response
        result = self._parse_forecast_response(data, latitude, longitude)
        
        # Update cache (Redis + memory)
        if self.cache_enabled:
            self._cache[cache_key] = result
            if REDIS_AVAILABLE:
                await cache_set(redis_key, result.to_dict(), ttl=FORECAST_CACHE_TTL)
        
        logger.info(
            "Fetched %d-day forecast for lat=%.4f, lon=%.4f",
            len(result.forecasts), latitude, longitude
        )
        
        return result
    
    def _dict_to_result(self, data: Dict[str, Any]) -> ForecastResult:
        """Reconstruct ForecastResult from cached dict."""
        forecasts = []
        for f in data.get("forecasts", []):
            forecasts.append(WeatherForecast(
                date=date.fromisoformat(f["date"]),
                temperature_max=f["temperature_max"],
                temperature_min=f["temperature_min"],
                precipitation_sum=f.get("precipitation_sum", 0),
                rain_sum=f.get("rain_sum", 0),
                precipitation_probability=f.get("precipitation_probability"),
                weather_code=f.get("weather_code"),
                weather_description=f.get("weather_description"),
                wind_speed_max=f.get("wind_speed_max"),
                humidity_mean=f.get("humidity_mean"),
                pressure_mean=f.get("pressure_mean"),
                flood_probability=f.get("flood_probability", 0),
                risk_level=RiskLevel(f.get("risk_level", "safe")),
                confidence=f.get("confidence"),
            ))
        return ForecastResult(
            latitude=data["latitude"],
            longitude=data["longitude"],
            forecasts=forecasts,
            fetched_at=datetime.fromisoformat(data["fetched_at"]),
            model=data.get("model", "unknown"),
            elevation=data.get("elevation", 0),
        )
    
    def _parse_forecast_response(
        self,
        data: Dict[str, Any],
        latitude: float,
        longitude: float,
    ) -> ForecastResult:
        """Parse Open-Meteo forecast API response."""
        forecasts = []
        
        daily = data.get("daily", {})
        if not daily:
            return ForecastResult(
                latitude=latitude,
                longitude=longitude,
                forecasts=[],
                fetched_at=datetime.now(),
                model="unknown",
                elevation=data.get("elevation", 0),
            )
        
        dates = daily.get("time", [])
        weather_codes = daily.get("weather_code", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip_sum = daily.get("precipitation_sum", [])
        rain_sum = daily.get("rain_sum", [])
        precip_prob = daily.get("precipitation_probability_max", [])
        wind_speed = daily.get("wind_speed_10m_max", [])
        humidity = daily.get("relative_humidity_2m_mean", [])
        pressure = daily.get("surface_pressure_mean", [])
        
        elevation = data.get("elevation", 50.0)
        today = date.today()
        
        # Track antecedent rainfall for risk calculation
        prev_rain = [0.0, 0.0, 0.0]  # Last 3 days
        
        for i, date_str in enumerate(dates):
            try:
                forecast_date = date.fromisoformat(date_str)
                
                # Skip past dates, only return future
                if forecast_date < today:
                    continue
                
                # Only return configured number of days
                days_ahead = (forecast_date - today).days
                if days_ahead > RETURN_FORECAST_DAYS:
                    break
                
                # Extract values with safe defaults
                code = int(weather_codes[i]) if i < len(weather_codes) and weather_codes[i] is not None else 0
                t_max = float(temp_max[i]) if i < len(temp_max) and temp_max[i] is not None else 30.0
                t_min = float(temp_min[i]) if i < len(temp_min) and temp_min[i] is not None else 20.0
                
                # Use rain_sum if available, otherwise precipitation_sum
                rain = 0.0
                if i < len(rain_sum) and rain_sum[i] is not None:
                    rain = float(rain_sum[i])
                elif i < len(precip_sum) and precip_sum[i] is not None:
                    rain = float(precip_sum[i])
                
                prob = float(precip_prob[i]) if i < len(precip_prob) and precip_prob[i] is not None else 0.0
                wind = float(wind_speed[i]) if i < len(wind_speed) and wind_speed[i] is not None else 10.0
                hum = float(humidity[i]) if i < len(humidity) and humidity[i] is not None else 60.0
                press = float(pressure[i]) if i < len(pressure) and pressure[i] is not None else 1013.25
                
                # Calculate antecedent rainfall
                antecedent = sum(prev_rain)
                
                # Calculate flood probability
                inputs = FloodRiskInput(
                    rain_sum=rain,
                    temperature_max=t_max,
                    temperature_min=t_min,
                    humidity_mean=hum,
                    pressure_mean=press,
                    wind_speed_max=wind,
                    antecedent_rain_3d=antecedent,
                    elevation_m=elevation,
                )
                
                flood_result = self.flood_calculator.calculate(inputs)
                flood_prob = flood_result.probability
                
                # Adjust flood probability by precipitation confidence
                # If precip_prob is low, reduce flood risk slightly
                confidence_factor = 0.5 + (prob / 200.0)  # 0.5 to 1.0
                adjusted_flood_prob = flood_prob * confidence_factor
                
                forecast = WeatherForecast(
                    date=forecast_date,
                    latitude=latitude,
                    longitude=longitude,
                    rain_sum=rain,
                    temperature_max=t_max,
                    temperature_min=t_min,
                    precipitation_probability=prob,
                    wind_speed_max=wind,
                    weather_code=code,
                    flood_probability=adjusted_flood_prob,
                )
                
                forecasts.append(forecast)
                
                # Update rolling antecedent rain
                prev_rain.pop(0)
                prev_rain.append(rain)
                
            except (ValueError, IndexError) as e:
                logger.warning("Failed to parse forecast %d: %s", i, e)
                continue
        
        return ForecastResult(
            latitude=data.get("latitude", latitude),
            longitude=data.get("longitude", longitude),
            forecasts=forecasts,
            fetched_at=datetime.now(),
            model=data.get("generationtime_ms", "open-meteo"),
            elevation=elevation,
        )
    
    async def get_forecast_day(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> Optional[WeatherForecast]:
        """
        Get forecast for a specific future date.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            target_date: The date to get forecast for
            
        Returns:
            WeatherForecast or None if date is out of range
        """
        today = date.today()
        days_ahead = (target_date - today).days
        
        if days_ahead < 0:
            logger.warning("Cannot get forecast for past date %s", target_date)
            return None
        
        if days_ahead > MAX_FORECAST_DAYS:
            logger.warning("Date %s is beyond forecast range", target_date)
            return None
        
        result = await self.get_forecast(latitude, longitude)
        
        for forecast in result.forecasts:
            if forecast.date == target_date:
                return forecast
        
        return None
    
    async def clear_cache(self, redis_too: bool = True) -> int:
        """Clear all cached forecasts. Returns number cleared."""
        count = len(self._cache)
        self._cache.clear()
        
        if redis_too and REDIS_AVAILABLE:
            redis_count = await cache_clear_prefix("forecast:")
            logger.info("Cleared %d Redis forecast cache entries", redis_count)
            count += redis_count
        
        return count


# ═══════════════════════════════════════════════════════════════════════════
# Synchronous wrapper for non-async code
# ═══════════════════════════════════════════════════════════════════════════

def get_forecast_sync(
    latitude: float,
    longitude: float,
) -> ForecastResult:
    """
    Synchronous wrapper for getting forecast.
    
    Usage:
        from backend.app.calendar.forecast_service import get_forecast_sync
        
        result = get_forecast_sync(12.9245, 80.0880)
        for f in result.forecasts:
            print(f"{f.date}: {f.rain_sum}mm, {f.flood_probability*100:.0f}% flood risk")
    """
    service = ForecastService()
    
    async def _fetch():
        try:
            return await service.get_forecast(latitude, longitude)
        finally:
            await service.close()
    
    return asyncio.run(_fetch())


# ═══════════════════════════════════════════════════════════════════════════
# API Error Handling
# ═══════════════════════════════════════════════════════════════════════════
#
# How to handle Open-Meteo API errors:
#
# 1. HTTP 400 - Bad Request
#    - Invalid coordinates, parameters, or date range
#    - Log error, return user-friendly message
#    - Don't retry
#
# 2. HTTP 429 - Rate Limited
#    - Too many requests (10,000/day free tier)
#    - Implement exponential backoff
#    - Cache aggressively to reduce calls
#
# 3. HTTP 500/502/503 - Server Error
#    - Temporary API issues
#    - Retry with exponential backoff (max 3 attempts)
#    - Return cached data if available
#
# 4. Network Timeout
#    - Increase timeout (30s recommended)
#    - Retry once
#    - Return cached data if available
#
# 5. Invalid JSON Response
#    - Log for debugging
#    - Return error to user
#    - Don't cache
#
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import sys
    
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 12.9245
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0880
    
    print(f"Fetching 5-day forecast for ({lat}, {lon})")
    
    result = get_forecast_sync(lat, lon)
    
    print(f"\nForecast ({result.model}):")
    print("-" * 60)
    
    for f in result.forecasts:
        risk_emoji = {
            "safe": "🟢",
            "watch": "🔵",
            "warning": "🟡",
            "severe": "🔴",
            "extreme": "⭕",
        }.get(f.risk_level.value, "⚪")
        
        print(
            f"{f.date.isoformat()} | "
            f"Rain: {f.rain_sum:5.1f}mm | "
            f"Temp: {f.temperature_min:.0f}-{f.temperature_max:.0f}°C | "
            f"Flood: {f.flood_probability*100:4.0f}% {risk_emoji} | "
            f"{f.weather_description}"
        )
