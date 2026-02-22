"""
weather_features.py — Transform raw weather data into ML-ready feature vectors.

This module sits between ingestion (weather_service.py) and ML inference.
It takes a WeatherResult and produces a flat feature dictionary that can be
fed directly into scikit-learn, XGBoost, or any tabular model.

Pipeline:
    weather_service.fetch_weather()    → WeatherResult (raw + accumulated)
    weather_features.build_features()  → Dict[str, float]  (ML-ready)
    ml/model.predict(features)         → risk probability

Feature Categories:
    1. CURRENT    — snapshot values (temp, wind, pressure, humidity)
    2. RAINFALL   — accumulation windows (1h, 3h, 6h, 24h) + ratios
    3. TREND      — recent change rates (pressure drop, temp rise)
    4. TERRAIN    — elevation (from API) + derived flags
    5. TEMPORAL   — hour, month, monsoon flag
    6. COMPOSITE  — cross-feature interactions (rain × soil_moisture, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.app.ingestion.weather_service import WeatherResult


def build_features(
    result: WeatherResult,
    *,
    include_trends: bool = True,
    include_temporal: bool = True,
    include_composite: bool = True,
) -> Dict[str, float]:
    """
    Build a complete ML feature vector from a WeatherResult.

    Parameters
    ----------
    result : WeatherResult
        Output from fetch_weather(). Must have success=True.
    include_trends : bool
        Compute pressure/temperature trends from hourly data.
    include_temporal : bool
        Add hour-of-day, month, monsoon-season flags.
    include_composite : bool
        Add cross-feature interaction terms.

    Returns
    -------
    Dict[str, float]
        Flat feature dict, all values numeric. Example:
        {
            "temperature_c": 32.5,
            "rain_24hr": 120.0,
            "pressure_drop_3hr": -4.2,
            "hour_sin": 0.866,
            "is_monsoon": 1.0,
            "rain_x_soil": 42.0,
            ...
        }

    Raises
    ------
    ValueError
        If result.success is False.
    """
    if not result.success:
        raise ValueError(
            f"Cannot build features from failed fetch: {result.error_message}"
        )

    features = result.to_ml_features()  # Start with base features

    # --- Trend features (requires hourly arrays) ---
    if include_trends and result.hourly:
        features.update(_compute_trends(result))

    # --- Temporal features ---
    if include_temporal:
        features.update(_compute_temporal(result))

    # --- Composite interaction features ---
    if include_composite:
        features.update(_compute_composite(features))

    return features


# ---------------------------------------------------------------------------
# Trend features — recent rate-of-change
# ---------------------------------------------------------------------------

def _compute_trends(result: WeatherResult) -> Dict[str, float]:
    """
    Compute short-term trends from hourly time series.

    A rapidly dropping barometric pressure is a strong cyclone signal.
    A rapid temperature rise with no rain signals heatwave buildup.
    """
    trends: Dict[str, float] = {}
    hourly = result.hourly

    if not hourly or len(hourly.timestamps) < 6:
        return {
            "pressure_drop_3hr": 0.0,
            "pressure_drop_6hr": 0.0,
            "temp_change_3hr": 0.0,
            "temp_change_6hr": 0.0,
            "wind_change_3hr": 0.0,
        }

    # Find current index
    from backend.app.ingestion.weather_service import _find_current_index
    idx = _find_current_index(hourly.timestamps)

    # Pressure trends (negative = dropping = storm approaching)
    if idx >= 3:
        trends["pressure_drop_3hr"] = round(
            hourly.pressure[idx] - hourly.pressure[idx - 3], 2
        )
    else:
        trends["pressure_drop_3hr"] = 0.0

    if idx >= 6:
        trends["pressure_drop_6hr"] = round(
            hourly.pressure[idx] - hourly.pressure[idx - 6], 2
        )
    else:
        trends["pressure_drop_6hr"] = 0.0

    # Temperature trends
    if idx >= 3:
        trends["temp_change_3hr"] = round(
            hourly.temperature_2m[idx] - hourly.temperature_2m[idx - 3], 2
        )
    else:
        trends["temp_change_3hr"] = 0.0

    if idx >= 6:
        trends["temp_change_6hr"] = round(
            hourly.temperature_2m[idx] - hourly.temperature_2m[idx - 6], 2
        )
    else:
        trends["temp_change_6hr"] = 0.0

    # Wind acceleration
    if idx >= 3:
        trends["wind_change_3hr"] = round(
            hourly.wind_speed[idx] - hourly.wind_speed[idx - 3], 2
        )
    else:
        trends["wind_change_3hr"] = 0.0

    return trends


# ---------------------------------------------------------------------------
# Temporal features — time-based signals
# ---------------------------------------------------------------------------

def _compute_temporal(result: WeatherResult) -> Dict[str, float]:
    """
    Encode time-of-day and seasonality as numeric features.

    Uses sine/cosine encoding for cyclical features (hour, month)
    so that hour 23 and hour 0 are numerically close.
    """
    import math

    now = datetime.now(timezone.utc)

    hour = now.hour
    month = now.month

    # Cyclical encoding: hour (period=24), month (period=12)
    temporal: Dict[str, float] = {
        "hour_sin": round(math.sin(2 * math.pi * hour / 24), 4),
        "hour_cos": round(math.cos(2 * math.pi * hour / 24), 4),
        "month_sin": round(math.sin(2 * math.pi * month / 12), 4),
        "month_cos": round(math.cos(2 * math.pi * month / 12), 4),
    }

    # India monsoon season flag (June–September = SW monsoon, Oct–Dec = NE monsoon)
    temporal["is_sw_monsoon"] = 1.0 if 6 <= month <= 9 else 0.0
    temporal["is_ne_monsoon"] = 1.0 if 10 <= month <= 12 else 0.0
    temporal["is_monsoon"] = 1.0 if 6 <= month <= 12 else 0.0

    # Night / dawn flag (certain disasters behave differently at night)
    temporal["is_night"] = 1.0 if hour < 6 or hour >= 20 else 0.0

    return temporal


# ---------------------------------------------------------------------------
# Composite features — cross-variable interactions
# ---------------------------------------------------------------------------

def _compute_composite(features: Dict[str, float]) -> Dict[str, float]:
    """
    Create interaction terms that capture compound risk signals.

    Examples:
        rain × soil_moisture → high values mean saturated ground + more rain → flood
        wind × low_pressure → cyclone signature
        temperature × humidity → heat index approximation
    """
    composite: Dict[str, float] = {}

    rain_24 = features.get("rain_24hr", 0.0)
    soil = features.get("soil_moisture", 0.0)
    wind = features.get("wind_speed_kmh", 0.0)
    temp = features.get("temperature_c", 0.0)
    humidity = features.get("humidity_pct", 0.0)
    pressure = features.get("pressure_hpa", 1013.0)
    rain_1 = features.get("rain_1hr", 0.0)

    # Rain on saturated soil = high flood risk
    composite["rain_x_soil"] = round(rain_24 * soil, 4)

    # Wind × inverse pressure (low pressure amplifies wind impact)
    pressure_deficit = max(0.0, 1013.0 - pressure)
    composite["wind_x_pressure_deficit"] = round(wind * pressure_deficit, 4)

    # Simplified heat index (°C).
    # Uses the simple Steadman formula: HI = 0.5 × {T + 61.0 + (T−68)×1.2 + RH×0.094}
    # converted to Celsius. For T > 27°C and RH > 40%, this approximates
    # the "feels-like" temperature accounting for humidity.
    # A more detailed Rothfusz regression exists but is overkill here.
    if temp > 27 and humidity > 40:
        # Simple Steadman (convert T to °F, compute, convert back)
        t_f = temp * 9.0 / 5.0 + 32.0
        hi_f = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + humidity * 0.094)
        hi_c = (hi_f - 32.0) * 5.0 / 9.0
        composite["heat_index"] = round(hi_c, 2)
    else:
        composite["heat_index"] = round(temp, 2)

    # Flood compound score (simple weighted sum)
    composite["flood_compound"] = round(
        rain_24 * 0.4
        + rain_1 * 0.3
        + soil * 100 * 0.2
        + (100 - features.get("elevation_m", 0)) * 0.01 * 0.1,
        4,
    )

    return composite


# ---------------------------------------------------------------------------
# Utility: print features table (for debugging / notebooks)
# ---------------------------------------------------------------------------

def print_features(features: Dict[str, float], title: str = "ML Features") -> None:
    """Pretty-print a feature dict as an aligned table."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    max_key = max(len(k) for k in features)
    for key, val in features.items():
        print(f"  {key:<{max_key}}  =  {val}")
    print(f"{'─' * 50}")
    print(f"  Total features: {len(features)}")
