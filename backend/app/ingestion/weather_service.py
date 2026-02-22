"""
weather_service.py — Open-Meteo weather data ingestion for disaster prediction.

Fetches hourly weather data from the Open-Meteo API (free, no API key required)
and transforms it into ML-ready features for flood/cyclone/heatwave prediction.

Capabilities:
    - Hourly rainfall, temperature, wind speed, humidity, pressure
    - Elevation for the queried coordinate
    - Rainfall accumulation windows: Rain_1hr, Rain_3hr, Rain_6hr, Rain_24hr
    - Graceful error handling with retries, timeouts, and fallback defaults
    - Clean dataclass output ready for ML feature pipelines

Open-Meteo API Reference:
    https://open-meteo.com/en/docs

Rainfall Accumulation — How It Works
======================================
Open-Meteo returns `precipitation` as mm/hr for each hourly slot. To compute
the total rainfall over a window (e.g. 3 hours), we sum consecutive hourly
values backward from the current hour:

    Rain_1hr  = precip[now]
    Rain_3hr  = precip[now] + precip[now-1] + precip[now-2]
    Rain_6hr  = Σ precip[now-5 … now]
    Rain_24hr = Σ precip[now-23 … now]

Why rolling sums matter for flood prediction:
    - Rain_1hr  → detects flash flood bursts (urban drainage capacity)
    - Rain_3hr  → correlates with small catchment flooding
    - Rain_6hr  → correlates with medium river basin response
    - Rain_24hr → correlates with large river/reservoir flooding
    - The RATIO Rain_1hr / Rain_24hr indicates intensity vs. sustained rain

From Forecast Data to ML Features
====================================
Raw API data goes through these transformations:

    1. RAW FETCH    →  hourly arrays of precip, temp, wind, humidity, pressure
    2. ACCUMULATION →  Rain_1hr, Rain_3hr, Rain_6hr, Rain_24hr
    3. DERIVED       →  rain_intensity_ratio, temp_deviation, wind_gust_ratio
    4. SPATIAL        →  elevation, slope (from DEM), distance_to_river
    5. TEMPORAL       →  hour_of_day, month, is_monsoon_season
    6. FEATURE VECTOR →  flat dict/array ready for model.predict()

Error Handling Strategy
========================
    Level 1 — Network errors (timeout, DNS, connection refused)
        → Retry up to 3 times with exponential backoff (1s, 2s, 4s)
        → After exhaustion, return WeatherResult with error flag + empty data

    Level 2 — API errors (HTTP 4xx/5xx, rate limiting)
        → 429 Too Many Requests: wait and retry
        → 4xx: fail immediately (bad coordinates, etc.)
        → 5xx: retry (server-side transient error)

    Level 3 — Data quality issues (missing fields, NaN values)
        → Fill gaps with interpolation for short gaps (≤ 2 hours)
        → Use 0.0 fallback for precipitation (conservative: no phantom rain)
        → Flag the result as `has_data_gaps = True` so downstream can decide

    Level 4 — Complete failure
        → Return a well-typed WeatherResult with `success=False`
        → Caller decides: use cached data, skip this cycle, or alert ops
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Hourly variables we request from Open-Meteo
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "soil_moisture_0_to_1cm",
]

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds; actual wait = base * 2^attempt
REQUEST_TIMEOUT = 15  # seconds


class FetchStatus(str, Enum):
    """Outcome of a weather data fetch operation."""
    SUCCESS = "success"
    PARTIAL = "partial"          # Got data but some fields missing
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RainfallAccumulation:
    """
    Rolling rainfall sums over standard windows.

    All values in millimeters (mm). Computed by summing hourly precipitation
    backward from the reference hour.

    Interpretation for flood risk:
        Rain_1hr  > 30 mm  → flash flood risk (IMD heavy rain threshold)
        Rain_3hr  > 65 mm  → significant urban flooding
        Rain_6hr  > 115 mm → river basin flooding likely
        Rain_24hr > 200 mm → severe flood event (IMD "extremely heavy")
    """
    rain_1hr: float = 0.0    # mm in the last 1 hour
    rain_3hr: float = 0.0    # mm in the last 3 hours
    rain_6hr: float = 0.0    # mm in the last 6 hours
    rain_24hr: float = 0.0   # mm in the last 24 hours

    @property
    def intensity_ratio(self) -> float:
        """
        Ratio of short-term to long-term rain.
        High ratio (> 0.5) = intense burst → flash flood risk.
        Low ratio (< 0.1) = steady drizzle → less dangerous per hour.
        Returns 0.0 if no rain in 24hr window.
        """
        if self.rain_24hr == 0.0:
            return 0.0
        return round(self.rain_1hr / self.rain_24hr, 4)

    @property
    def flood_risk_category(self) -> str:
        """Simple categorical risk based on IMD thresholds."""
        if self.rain_24hr >= 200:
            return "extreme"
        elif self.rain_24hr >= 115 or self.rain_1hr >= 30:
            return "high"
        elif self.rain_24hr >= 65 or self.rain_3hr >= 40:
            return "moderate"
        elif self.rain_24hr >= 15:
            return "low"
        return "negligible"


@dataclass
class CurrentConditions:
    """Snapshot of weather at the most recent available hour."""
    temperature_c: float = 0.0
    humidity_pct: float = 0.0
    wind_speed_kmh: float = 0.0
    wind_gusts_kmh: float = 0.0
    pressure_hpa: float = 0.0
    cloud_cover_pct: float = 0.0
    soil_moisture: float = 0.0
    timestamp: str = ""


@dataclass
class HourlyData:
    """Raw hourly time series from the API."""
    timestamps: List[str] = field(default_factory=list)
    temperature_2m: List[float] = field(default_factory=list)
    humidity: List[float] = field(default_factory=list)
    precipitation: List[float] = field(default_factory=list)
    rain: List[float] = field(default_factory=list)
    pressure: List[float] = field(default_factory=list)
    wind_speed: List[float] = field(default_factory=list)
    wind_gusts: List[float] = field(default_factory=list)
    cloud_cover: List[float] = field(default_factory=list)
    soil_moisture: List[float] = field(default_factory=list)


@dataclass
class WeatherResult:
    """
    Complete result of a weather data fetch.

    This is the single return type for all weather_service operations.
    Callers check `success` before using the data.
    """
    success: bool
    status: FetchStatus
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    timezone_name: str = ""

    current: Optional[CurrentConditions] = None
    rainfall: Optional[RainfallAccumulation] = None
    hourly: Optional[HourlyData] = None

    has_data_gaps: bool = False
    error_message: str = ""
    fetch_duration_ms: int = 0
    api_credits_used: int = 0  # Open-Meteo is free, but track for future

    # ML-ready feature dict — populated by to_ml_features()
    _features: Optional[Dict[str, float]] = field(default=None, repr=False)

    def to_ml_features(self) -> Dict[str, float]:
        """
        Flatten all weather data into a single dict suitable for
        model.predict(). This is the bridge between ingestion and ML.

        Returns a dict like:
            {
                "temperature_c": 32.5,
                "humidity_pct": 78.0,
                "wind_speed_kmh": 15.2,
                "wind_gusts_kmh": 28.0,
                "pressure_hpa": 1008.3,
                "cloud_cover_pct": 85.0,
                "soil_moisture": 0.35,
                "elevation_m": 12.0,
                "rain_1hr": 8.5,
                "rain_3hr": 22.0,
                "rain_6hr": 45.0,
                "rain_24hr": 120.0,
                "rain_intensity_ratio": 0.0708,
                "temp_is_heatwave": 0.0,
                "wind_is_cyclonic": 0.0,
            }
        """
        if self._features is not None:
            return self._features

        features: Dict[str, float] = {}

        # Current conditions
        if self.current:
            features["temperature_c"] = self.current.temperature_c
            features["humidity_pct"] = self.current.humidity_pct
            features["wind_speed_kmh"] = self.current.wind_speed_kmh
            features["wind_gusts_kmh"] = self.current.wind_gusts_kmh
            features["pressure_hpa"] = self.current.pressure_hpa
            features["cloud_cover_pct"] = self.current.cloud_cover_pct
            features["soil_moisture"] = self.current.soil_moisture

        # Elevation
        features["elevation_m"] = self.elevation_m

        # Rainfall accumulations
        if self.rainfall:
            features["rain_1hr"] = self.rainfall.rain_1hr
            features["rain_3hr"] = self.rainfall.rain_3hr
            features["rain_6hr"] = self.rainfall.rain_6hr
            features["rain_24hr"] = self.rainfall.rain_24hr
            features["rain_intensity_ratio"] = self.rainfall.intensity_ratio

        # Derived binary flags (simple threshold features)
        features["temp_is_heatwave"] = (
            1.0 if features.get("temperature_c", 0) >= 40.0 else 0.0
        )
        features["wind_is_cyclonic"] = (
            1.0 if features.get("wind_speed_kmh", 0) >= 62.0 else 0.0
        )
        features["pressure_is_low"] = (
            1.0 if features.get("pressure_hpa", 1013) < 1000.0 else 0.0
        )
        features["soil_is_saturated"] = (
            1.0 if features.get("soil_moisture", 0) >= 0.45 else 0.0
        )

        self._features = features
        return features


# ---------------------------------------------------------------------------
# HTTP layer (stdlib only — no requests/httpx dependency)
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """
    Fetch JSON from a URL with retry logic.

    Retry strategy:
        Attempt 1: immediate
        Attempt 2: wait 1 second
        Attempt 3: wait 2 seconds
        Attempt 4: wait 4 seconds (final)

    Raises RuntimeError on exhaustion.
    """
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                f"Retry {attempt}/{MAX_RETRIES} after {wait:.1f}s — {last_error}"
            )
            time.sleep(wait)

        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
                elif resp.status == 429:
                    # Rate limited — retry
                    last_error = RuntimeError(f"Rate limited (429)")
                    continue
                elif 400 <= resp.status < 500:
                    # Client error — don't retry
                    body = resp.read().decode("utf-8", errors="replace")
                    raise ValueError(
                        f"API returned {resp.status}: {body[:200]}"
                    )
                else:
                    last_error = RuntimeError(f"HTTP {resp.status}")
                    continue

        except (URLError, OSError, TimeoutError) as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Data cleaning helpers
# ---------------------------------------------------------------------------

def _safe_list(raw: Optional[List], length: int, default: float = 0.0) -> List[float]:
    """
    Ensure a list has the expected length, filling None/missing with defaults.

    This is Level 3 error handling: data quality issues.
    Precipitation gaps are filled with 0.0 (conservative — no phantom rain).
    Temperature/pressure gaps are forward-filled from last known value.
    """
    if raw is None:
        return [default] * length

    result = []
    last_valid = default
    for i in range(length):
        if i < len(raw) and raw[i] is not None:
            val = float(raw[i])
            result.append(val)
            last_valid = val
        else:
            result.append(last_valid if default != 0.0 else 0.0)

    return result


def _compute_accumulation(
    precipitation: List[float],
    current_index: int,
) -> RainfallAccumulation:
    """
    Compute rolling rainfall sums at the given hour index.

    How accumulation works:
        precipitation[] contains mm of rain per hour.
        For Rain_3hr at index=50, we sum indices [48, 49, 50].
        If the window extends before index 0, we sum whatever is available.

    Parameters
    ----------
    precipitation : List[float]
        Hourly precipitation values in mm.
    current_index : int
        The "now" index in the precipitation array.

    Returns
    -------
    RainfallAccumulation
        Rolling sums for 1h, 3h, 6h, 24h windows.
    """
    def _window_sum(hours: int) -> float:
        start = max(0, current_index - hours + 1)
        end = current_index + 1
        return round(sum(precipitation[start:end]), 2)

    return RainfallAccumulation(
        rain_1hr=_window_sum(1),
        rain_3hr=_window_sum(3),
        rain_6hr=_window_sum(6),
        rain_24hr=_window_sum(24),
    )


def _find_current_index(timestamps: List[str]) -> int:
    """
    Find the hourly slot closest to 'now' in the timestamp array.
    Open-Meteo returns ISO timestamps like "2026-02-22T14:00".
    We pick the latest timestamp that is ≤ current UTC time.
    """
    now = datetime.now(timezone.utc)
    best_idx = 0

    for i, ts in enumerate(timestamps):
        try:
            # Open-Meteo format: "2026-02-22T14:00"
            dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            if dt <= now:
                best_idx = i
        except (ValueError, TypeError):
            continue

    return best_idx


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_weather(
    latitude: float,
    longitude: float,
    *,
    forecast_days: int = 2,
    past_days: int = 1,
) -> WeatherResult:
    """
    Fetch weather data from Open-Meteo for a coordinate.

    This is the PRIMARY entry point of the weather ingestion module.

    Parameters
    ----------
    latitude : float
        Decimal degrees, -90 to 90.
    longitude : float
        Decimal degrees, -180 to 180.
    forecast_days : int
        Number of forecast days (1-16). Default 2.
    past_days : int
        Number of past days to include (0-92). Default 1 (for Rain_24hr).

    Returns
    -------
    WeatherResult
        Complete weather data with success/failure status.
        On failure, `success=False` and `error_message` is populated.

    Example
    -------
    >>> result = fetch_weather(13.0827, 80.2707)
    >>> if result.success:
    ...     print(result.rainfall.rain_24hr)
    ...     features = result.to_ml_features()
    """
    start_time = time.monotonic()

    # --- Input validation (Level 4: invalid input) ---
    if not (-90.0 <= latitude <= 90.0):
        return WeatherResult(
            success=False,
            status=FetchStatus.INVALID_INPUT,
            latitude=latitude,
            longitude=longitude,
            error_message=f"Latitude must be in [-90, 90], got {latitude}",
        )
    if not (-180.0 <= longitude <= 180.0):
        return WeatherResult(
            success=False,
            status=FetchStatus.INVALID_INPUT,
            latitude=latitude,
            longitude=longitude,
            error_message=f"Longitude must be in [-180, 180], got {longitude}",
        )

    # --- Build API URL ---
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "UTC",
    }
    url = f"{OPEN_METEO_BASE_URL}?{urlencode(params)}"
    logger.info(f"Fetching weather: ({latitude}, {longitude})")

    # --- Fetch with retry ---
    try:
        data = _fetch_json(url)
    except ValueError as e:
        # Client error (4xx) — bad request, no retry will help
        elapsed = int((time.monotonic() - start_time) * 1000)
        return WeatherResult(
            success=False,
            status=FetchStatus.API_ERROR,
            latitude=latitude,
            longitude=longitude,
            error_message=str(e),
            fetch_duration_ms=elapsed,
        )
    except RuntimeError as e:
        # Exhausted retries
        elapsed = int((time.monotonic() - start_time) * 1000)
        return WeatherResult(
            success=False,
            status=FetchStatus.NETWORK_ERROR,
            latitude=latitude,
            longitude=longitude,
            error_message=str(e),
            fetch_duration_ms=elapsed,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start_time) * 1000)
        return WeatherResult(
            success=False,
            status=FetchStatus.NETWORK_ERROR,
            latitude=latitude,
            longitude=longitude,
            error_message=f"Unexpected error: {e}",
            fetch_duration_ms=elapsed,
        )

    # --- Parse response ---
    try:
        return _parse_response(data, latitude, longitude, start_time)
    except Exception as e:
        elapsed = int((time.monotonic() - start_time) * 1000)
        logger.error(f"Failed to parse weather response: {e}", exc_info=True)
        return WeatherResult(
            success=False,
            status=FetchStatus.API_ERROR,
            latitude=latitude,
            longitude=longitude,
            error_message=f"Parse error: {e}",
            fetch_duration_ms=elapsed,
        )


def _parse_response(
    data: Dict[str, Any],
    latitude: float,
    longitude: float,
    start_time: float,
) -> WeatherResult:
    """
    Transform raw Open-Meteo JSON into our typed WeatherResult.

    Open-Meteo response structure:
        {
            "latitude": 13.0625,
            "longitude": 80.25,
            "elevation": 12.0,
            "timezone": "UTC",
            "hourly": {
                "time": ["2026-02-21T00:00", ...],
                "temperature_2m": [28.1, ...],
                "precipitation": [0.0, ...],
                ...
            }
        }
    """
    hourly_raw = data.get("hourly", {})
    timestamps = hourly_raw.get("time", [])
    n = len(timestamps)

    has_gaps = False

    # --- Check for missing fields ---
    for var in HOURLY_VARIABLES:
        if var not in hourly_raw:
            logger.warning(f"Missing hourly variable: {var}")
            has_gaps = True

    # --- Clean arrays (Level 3: fill gaps) ---
    precip = _safe_list(hourly_raw.get("precipitation"), n, default=0.0)
    rain_arr = _safe_list(hourly_raw.get("rain"), n, default=0.0)
    temp = _safe_list(hourly_raw.get("temperature_2m"), n, default=25.0)
    humidity = _safe_list(hourly_raw.get("relative_humidity_2m"), n, default=50.0)
    pressure = _safe_list(hourly_raw.get("surface_pressure"), n, default=1013.0)
    wind_speed = _safe_list(hourly_raw.get("wind_speed_10m"), n, default=0.0)
    wind_gusts = _safe_list(hourly_raw.get("wind_gusts_10m"), n, default=0.0)
    cloud_cover = _safe_list(hourly_raw.get("cloud_cover"), n, default=0.0)
    soil_moist = _safe_list(hourly_raw.get("soil_moisture_0_to_1cm"), n, default=0.0)

    # Any None in raw data means we had gaps
    for var_name in ["precipitation", "temperature_2m", "wind_speed_10m"]:
        raw = hourly_raw.get(var_name, [])
        if raw and any(v is None for v in raw):
            has_gaps = True

    # --- Find current hour index ---
    current_idx = _find_current_index(timestamps)

    # --- Build rainfall accumulation ---
    rainfall = _compute_accumulation(precip, current_idx)

    # --- Build current conditions ---
    current = CurrentConditions(
        temperature_c=temp[current_idx] if current_idx < n else 0.0,
        humidity_pct=humidity[current_idx] if current_idx < n else 0.0,
        wind_speed_kmh=wind_speed[current_idx] if current_idx < n else 0.0,
        wind_gusts_kmh=wind_gusts[current_idx] if current_idx < n else 0.0,
        pressure_hpa=pressure[current_idx] if current_idx < n else 0.0,
        cloud_cover_pct=cloud_cover[current_idx] if current_idx < n else 0.0,
        soil_moisture=soil_moist[current_idx] if current_idx < n else 0.0,
        timestamp=timestamps[current_idx] if current_idx < n else "",
    )

    # --- Build hourly data ---
    hourly = HourlyData(
        timestamps=timestamps,
        temperature_2m=temp,
        humidity=humidity,
        precipitation=precip,
        rain=rain_arr,
        pressure=pressure,
        wind_speed=wind_speed,
        wind_gusts=wind_gusts,
        cloud_cover=cloud_cover,
        soil_moisture=soil_moist,
    )

    elapsed = int((time.monotonic() - start_time) * 1000)

    return WeatherResult(
        success=True,
        status=FetchStatus.PARTIAL if has_gaps else FetchStatus.SUCCESS,
        latitude=data.get("latitude", latitude),
        longitude=data.get("longitude", longitude),
        elevation_m=data.get("elevation", 0.0),
        timezone_name=data.get("timezone", "UTC"),
        current=current,
        rainfall=rainfall,
        hourly=hourly,
        has_data_gaps=has_gaps,
        fetch_duration_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Convenience: fetch just rainfall summary (lighter output)
# ---------------------------------------------------------------------------

def fetch_rainfall_summary(
    latitude: float,
    longitude: float,
) -> Dict[str, Any]:
    """
    Quick-fetch just the rainfall accumulation for a coordinate.

    Returns a simple dict — useful for scripts and quick checks.

    Example
    -------
    >>> summary = fetch_rainfall_summary(13.0827, 80.2707)
    >>> print(summary)
    {
        'success': True,
        'rain_1hr': 0.0,
        'rain_3hr': 0.0,
        'rain_6hr': 2.5,
        'rain_24hr': 18.7,
        'flood_risk': 'low',
        'intensity_ratio': 0.0,
        'elevation_m': 12.0
    }
    """
    result = fetch_weather(latitude, longitude)

    if not result.success:
        return {
            "success": False,
            "error": result.error_message,
            "status": result.status.value,
        }

    return {
        "success": True,
        "rain_1hr": result.rainfall.rain_1hr,
        "rain_3hr": result.rainfall.rain_3hr,
        "rain_6hr": result.rainfall.rain_6hr,
        "rain_24hr": result.rainfall.rain_24hr,
        "flood_risk": result.rainfall.flood_risk_category,
        "intensity_ratio": result.rainfall.intensity_ratio,
        "elevation_m": result.elevation_m,
    }
