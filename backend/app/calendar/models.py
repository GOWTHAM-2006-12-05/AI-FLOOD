"""
Database models for Calendar Weather System.

═══════════════════════════════════════════════════════════════════════════
DATABASE SCHEMA DESIGN
═══════════════════════════════════════════════════════════════════════════

Table: historical_weather
─────────────────────────────────────────────────────────────────────────────
| Column            | Type          | Description                          |
|-------------------|---------------|--------------------------------------|
| id                | SERIAL PK     | Auto-increment primary key           |
| latitude          | DECIMAL(9,6)  | Location latitude (-90 to 90)        |
| longitude         | DECIMAL(10,6) | Location longitude (-180 to 180)     |
| date              | DATE          | Weather date                         |
| rain_sum          | DECIMAL(8,2)  | Daily rainfall in mm                 |
| temperature_max   | DECIMAL(5,2)  | Max temperature °C                   |
| temperature_min   | DECIMAL(5,2)  | Min temperature °C                   |
| humidity_mean     | DECIMAL(5,2)  | Mean humidity %                      |
| pressure_mean     | DECIMAL(7,2)  | Mean pressure hPa                    |
| wind_speed_max    | DECIMAL(6,2)  | Max wind speed km/h                  |
| flood_probability | DECIMAL(5,4)  | Computed flood risk (0-1)            |
| created_at        | TIMESTAMP     | Record creation time                 |
| updated_at        | TIMESTAMP     | Last update time                     |
─────────────────────────────────────────────────────────────────────────────

Constraints:
- UNIQUE (latitude, longitude, date) - prevents duplicate entries
- INDEX on date for fast date queries
- INDEX on (latitude, longitude) for location queries

Query Patterns:
1. Specific date: WHERE date = '2025-06-15' AND latitude = X AND longitude = Y
2. Full year: WHERE EXTRACT(YEAR FROM date) = 2025
3. Date range: WHERE date BETWEEN '2025-01-01' AND '2025-12-31'

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskLevel(str, Enum):
    """Flood risk classification levels."""
    SAFE = "safe"           # Green - No concern
    WATCH = "watch"         # Blue - Monitor situation
    WARNING = "warning"     # Yellow/Orange - Prepare
    SEVERE = "severe"       # Red - Immediate action
    EXTREME = "extreme"     # Dark Red - Emergency


@dataclass
class HistoricalWeather:
    """
    Historical weather data record.
    
    Stored in PostgreSQL for fast retrieval of past weather data.
    Used for:
    - Calendar display of past dates
    - ML model training data
    - Trend analysis
    """
    id: Optional[int] = None
    latitude: float = 0.0
    longitude: float = 0.0
    date: date = field(default_factory=date.today)
    rain_sum: float = 0.0
    temperature_max: float = 0.0
    temperature_min: float = 0.0
    humidity_mean: float = 0.0
    pressure_mean: float = 1013.25
    wind_speed_max: float = 0.0
    flood_probability: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def risk_level(self) -> RiskLevel:
        """Derive risk level from flood probability."""
        if self.flood_probability >= 0.8:
            return RiskLevel.EXTREME
        elif self.flood_probability >= 0.6:
            return RiskLevel.SEVERE
        elif self.flood_probability >= 0.35:
            return RiskLevel.WARNING
        elif self.flood_probability >= 0.15:
            return RiskLevel.WATCH
        else:
            return RiskLevel.SAFE
    
    @property
    def risk_color(self) -> str:
        """CSS color for calendar cell."""
        colors = {
            RiskLevel.SAFE: "#22c55e",      # Green
            RiskLevel.WATCH: "#3b82f6",     # Blue
            RiskLevel.WARNING: "#f59e0b",   # Orange
            RiskLevel.SEVERE: "#ef4444",    # Red
            RiskLevel.EXTREME: "#991b1b",   # Dark red
        }
        return colors.get(self.risk_level, "#6b7280")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": self.id,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "date": self.date.isoformat(),
            "rain_mm": round(self.rain_sum, 2),
            "temperature_max": round(self.temperature_max, 1),
            "temperature_min": round(self.temperature_min, 1),
            "humidity_mean": round(self.humidity_mean, 1),
            "pressure_mean": round(self.pressure_mean, 1),
            "wind_speed_max": round(self.wind_speed_max, 1),
            "flood_probability": round(self.flood_probability * 100, 1),
            "risk_level": self.risk_level.value,
            "risk_color": self.risk_color,
        }


@dataclass
class WeatherForecast:
    """
    Weather forecast for future dates.
    
    Fetched from Open-Meteo Forecast API.
    Not stored in database (ephemeral).
    """
    date: date
    latitude: float
    longitude: float
    rain_sum: float = 0.0
    temperature_max: float = 0.0
    temperature_min: float = 0.0
    precipitation_probability: float = 0.0
    wind_speed_max: float = 0.0
    weather_code: int = 0  # WMO weather code
    flood_probability: float = 0.0
    
    @property
    def risk_level(self) -> RiskLevel:
        """Derive risk level from flood probability."""
        if self.flood_probability >= 0.8:
            return RiskLevel.EXTREME
        elif self.flood_probability >= 0.6:
            return RiskLevel.SEVERE
        elif self.flood_probability >= 0.35:
            return RiskLevel.WARNING
        elif self.flood_probability >= 0.15:
            return RiskLevel.WATCH
        else:
            return RiskLevel.SAFE
    
    @property
    def risk_color(self) -> str:
        """CSS color for calendar cell."""
        colors = {
            RiskLevel.SAFE: "#22c55e",
            RiskLevel.WATCH: "#3b82f6",
            RiskLevel.WARNING: "#f59e0b",
            RiskLevel.SEVERE: "#ef4444",
            RiskLevel.EXTREME: "#991b1b",
        }
        return colors.get(self.risk_level, "#6b7280")
    
    @property
    def weather_description(self) -> str:
        """Human-readable weather from WMO code."""
        wmo_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail",
        }
        return wmo_codes.get(self.weather_code, "Unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "date": self.date.isoformat(),
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "rain_mm": round(self.rain_sum, 2),
            "temperature_max": round(self.temperature_max, 1),
            "temperature_min": round(self.temperature_min, 1),
            "precipitation_probability": round(self.precipitation_probability, 1),
            "wind_speed_max": round(self.wind_speed_max, 1),
            "weather_code": self.weather_code,
            "weather_description": self.weather_description,
            "flood_probability": round(self.flood_probability * 100, 1),
            "risk_level": self.risk_level.value,
            "risk_color": self.risk_color,
        }


@dataclass 
class CalendarDayData:
    """
    Combined data for a single calendar day.
    
    Can be either historical or forecast data.
    """
    date: date
    is_forecast: bool  # True if future date
    is_today: bool
    rain_mm: float
    temperature_max: float
    temperature_min: float
    flood_probability: float  # 0-100
    risk_level: str
    risk_color: str
    weather_description: str = ""
    has_data: bool = True  # False if no data available
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "is_forecast": self.is_forecast,
            "is_today": self.is_today,
            "rain_mm": round(self.rain_mm, 2),
            "temperature_max": round(self.temperature_max, 1),
            "temperature_min": round(self.temperature_min, 1),
            "flood_probability": round(self.flood_probability, 1),
            "risk_level": self.risk_level,
            "risk_color": self.risk_color,
            "weather_description": self.weather_description,
            "has_data": self.has_data,
        }


@dataclass
class CalendarMonthData:
    """
    Full month of calendar data.
    """
    year: int
    month: int
    latitude: float
    longitude: float
    days: List[CalendarDayData]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "month": self.month,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "days": [d.to_dict() for d in self.days],
            "summary": {
                "total_rain_mm": round(sum(d.rain_mm for d in self.days if d.has_data), 2),
                "avg_temp_max": round(
                    sum(d.temperature_max for d in self.days if d.has_data) / 
                    max(1, len([d for d in self.days if d.has_data])), 1
                ),
                "high_risk_days": len([d for d in self.days if d.flood_probability >= 35]),
                "severe_days": len([d for d in self.days if d.flood_probability >= 60]),
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# SQL Schema for PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════

SQL_CREATE_SCHEMA = """
-- Historical Weather Data Table
CREATE TABLE IF NOT EXISTS historical_weather (
    id SERIAL PRIMARY KEY,
    latitude DECIMAL(9, 6) NOT NULL,
    longitude DECIMAL(10, 6) NOT NULL,
    date DATE NOT NULL,
    rain_sum DECIMAL(8, 2) DEFAULT 0,
    temperature_max DECIMAL(5, 2),
    temperature_min DECIMAL(5, 2),
    humidity_mean DECIMAL(5, 2),
    pressure_mean DECIMAL(7, 2) DEFAULT 1013.25,
    wind_speed_max DECIMAL(6, 2) DEFAULT 0,
    flood_probability DECIMAL(5, 4) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent duplicate entries for same location/date
    CONSTRAINT unique_location_date UNIQUE (latitude, longitude, date)
);

-- Index for fast date queries
CREATE INDEX IF NOT EXISTS idx_historical_weather_date 
    ON historical_weather(date);

-- Index for location-based queries
CREATE INDEX IF NOT EXISTS idx_historical_weather_location 
    ON historical_weather(latitude, longitude);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_historical_weather_location_date 
    ON historical_weather(latitude, longitude, date);

-- Index for year-based queries
CREATE INDEX IF NOT EXISTS idx_historical_weather_year 
    ON historical_weather(EXTRACT(YEAR FROM date));

-- Ingestion tracking table
CREATE TABLE IF NOT EXISTS weather_ingestion_log (
    id SERIAL PRIMARY KEY,
    latitude DECIMAL(9, 6) NOT NULL,
    longitude DECIMAL(10, 6) NOT NULL,
    year INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'pending', 'in_progress', 'completed', 'failed'
    records_inserted INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_location_year UNIQUE (latitude, longitude, year)
);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_historical_weather_updated_at
    BEFORE UPDATE ON historical_weather
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""

SQL_QUERY_SPECIFIC_DATE = """
SELECT * FROM historical_weather 
WHERE latitude = %s AND longitude = %s AND date = %s;
"""

SQL_QUERY_FULL_YEAR = """
SELECT * FROM historical_weather 
WHERE latitude = %s AND longitude = %s 
  AND EXTRACT(YEAR FROM date) = %s
ORDER BY date;
"""

SQL_QUERY_DATE_RANGE = """
SELECT * FROM historical_weather 
WHERE latitude = %s AND longitude = %s 
  AND date BETWEEN %s AND %s
ORDER BY date;
"""

SQL_QUERY_MONTH = """
SELECT * FROM historical_weather 
WHERE latitude = %s AND longitude = %s 
  AND EXTRACT(YEAR FROM date) = %s
  AND EXTRACT(MONTH FROM date) = %s
ORDER BY date;
"""

SQL_UPSERT_WEATHER = """
INSERT INTO historical_weather 
    (latitude, longitude, date, rain_sum, temperature_max, temperature_min, 
     humidity_mean, pressure_mean, wind_speed_max, flood_probability)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (latitude, longitude, date) 
DO UPDATE SET
    rain_sum = EXCLUDED.rain_sum,
    temperature_max = EXCLUDED.temperature_max,
    temperature_min = EXCLUDED.temperature_min,
    humidity_mean = EXCLUDED.humidity_mean,
    pressure_mean = EXCLUDED.pressure_mean,
    wind_speed_max = EXCLUDED.wind_speed_max,
    flood_probability = EXCLUDED.flood_probability,
    updated_at = CURRENT_TIMESTAMP;
"""
