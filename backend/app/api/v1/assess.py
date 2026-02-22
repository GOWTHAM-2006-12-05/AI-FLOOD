"""
FastAPI route: Unified Location Risk Assessment.

This endpoint orchestrates the FULL pipeline:
    1. Fetch live weather from Open-Meteo
    2. Run flood prediction (weather → features → XGBoost+LSTM ensemble)
    3. Fetch earthquakes from USGS
    4. Detect cyclone conditions from weather data
    5. Aggregate all hazard scores into unified risk score
    6. Return complete RiskData for the frontend dashboard

This is the endpoint the frontend dashboard should call.
"""

from __future__ import annotations

import logging
import traceback
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE REGIONAL RISK DATA
# ═══════════════════════════════════════════════════════════════════════════

# India Seismic Zones (simplified polygons)
# Zone V: Very High Risk (Northeast, Kashmir, Gujarat Rann)
# Zone IV: High Risk (Delhi, Mumbai, parts of HP, Bihar)
# Zone III: Moderate Risk (Chennai, Bangalore, most of peninsula)
# Zone II: Low Risk (Central plateau)
# Reference: Bureau of Indian Standards IS:1893

SEISMIC_ZONES = [
    # (name, center_lat, center_lon, radius_km, zone_level, base_risk)
    # Zone V - Very High
    ("Northeast India", 26.2, 92.0, 300, 5, 0.25),
    ("Kashmir", 34.0, 75.0, 200, 5, 0.25),
    ("Andaman Islands", 12.0, 93.0, 200, 5, 0.25),
    ("Gujarat Kutch", 23.5, 70.0, 150, 5, 0.25),
    
    # Zone IV - High
    ("Delhi NCR", 28.6, 77.2, 100, 4, 0.15),
    ("Himachal Pradesh", 31.5, 77.0, 150, 4, 0.15),
    ("Northern Bihar", 26.5, 85.5, 150, 4, 0.15),
    ("Uttarakhand", 30.0, 79.0, 150, 4, 0.15),
    ("Mumbai Region", 19.0, 72.8, 80, 4, 0.12),
    ("Pakistan Border Punjab", 31.5, 74.5, 100, 4, 0.12),
    
    # Zone III - Moderate (covers most of India)
    ("Maharashtra", 19.0, 76.0, 300, 3, 0.08),
    ("Kerala Coast", 10.0, 76.2, 150, 3, 0.08),
    ("Tamil Nadu Coast", 11.5, 79.5, 200, 3, 0.08),
    ("Odisha Coast", 20.5, 85.5, 200, 3, 0.08),
    ("West Bengal", 23.0, 88.0, 150, 3, 0.08),
    ("Rajasthan", 27.0, 74.0, 250, 3, 0.06),
    ("Madhya Pradesh", 23.0, 78.0, 300, 3, 0.05),
    
    # Default Zone II - Low (interior plateau)
    ("Central India Default", 20.0, 78.0, 800, 2, 0.03),
]

# Major cyclone-prone coastal areas (Bay of Bengal & Arabian Sea)
CYCLONE_ZONES = [
    # (name, center_lat, center_lon, radius_km, base_risk_peak_season, season_months)
    # Bay of Bengal - East Coast (primary cyclone track)
    ("Odisha Coast", 20.0, 86.5, 200, 0.20, [4, 5, 10, 11, 12]),
    ("Andhra Coast", 16.0, 81.5, 200, 0.18, [4, 5, 10, 11, 12]),
    ("Tamil Nadu Coast", 12.0, 80.0, 150, 0.15, [10, 11, 12]),
    ("West Bengal Coast", 22.0, 88.5, 150, 0.15, [4, 5, 10, 11]),
    ("Bangladesh Border", 22.5, 91.0, 150, 0.18, [4, 5, 10, 11]),
    
    # Arabian Sea - West Coast (less frequent but intense)
    ("Gujarat Coast", 21.5, 70.0, 200, 0.12, [5, 6, 10, 11]),
    ("Maharashtra Coast", 18.0, 73.0, 150, 0.08, [5, 6, 10, 11]),
    ("Goa-Karnataka Coast", 15.0, 74.0, 150, 0.06, [5, 6, 10, 11]),
    ("Kerala Coast", 9.5, 76.0, 150, 0.05, [5, 6, 10, 11]),
    
    # Andaman & Nicobar
    ("Andaman Islands", 12.0, 93.0, 200, 0.22, [4, 5, 10, 11, 12]),
]


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _get_seismic_zone_risk(lat: float, lon: float) -> Tuple[float, str, int]:
    """
    Get baseline earthquake risk based on seismic zone.
    
    Returns (base_risk, zone_name, zone_level)
    """
    best_match = ("Default", 0.02, 2)  # Default Zone II
    best_dist = float('inf')
    
    for name, clat, clon, radius, level, risk in SEISMIC_ZONES:
        dist = _haversine_distance(lat, lon, clat, clon)
        if dist <= radius:
            # Inside this zone - prefer higher risk zones
            if risk > best_match[1] or (risk == best_match[1] and dist < best_dist):
                best_match = (name, risk, level)
                best_dist = dist
    
    return best_match[1], best_match[0], best_match[2]


def _get_cyclone_zone_risk(lat: float, lon: float) -> Tuple[float, str, bool]:
    """
    Get baseline cyclone risk based on coastal zone and season.
    
    Returns (base_risk, zone_name, is_peak_season)
    """
    current_month = datetime.now().month
    best_match = (0.01, "Inland", False)  # Default for inland areas
    
    for name, clat, clon, radius, peak_risk, months in CYCLONE_ZONES:
        dist = _haversine_distance(lat, lon, clat, clon)
        if dist <= radius:
            is_peak = current_month in months
            # Risk is higher during peak season, lower otherwise
            if is_peak:
                risk = peak_risk
            else:
                risk = peak_risk * 0.4  # 40% of peak risk during off-season
            
            if risk > best_match[0]:
                best_match = (risk, name, is_peak)
    
    return best_match

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risk", tags=["risk-assessment"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AssessRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    radius_km: float = Field(50.0, gt=0, le=2000, description="Alert radius in km")


class ModelConfidenceOut(BaseModel):
    xgboost: float = 0.0
    lstm: float = 0.0
    ensemble: float = 0.0
    agreement: bool = True


class HazardBreakdownOut(BaseModel):
    hazard_type: str
    raw_value: float
    normalised_score: float
    weight: float
    weighted_contribution: float
    is_active: bool
    is_critical: bool
    priority: int


class FeatureImportanceOut(BaseModel):
    feature: str
    importance: float
    category: str


class AssessResponse(BaseModel):
    overall_risk_score: float
    overall_risk_level: str
    dominant_hazard: str
    alert_action: str
    alert_reasons: List[str]
    active_hazard_count: int
    ensemble_alpha: float = 0.65
    model_confidence: ModelConfidenceOut
    feature_importance: List[FeatureImportanceOut] = []
    hazard_breakdown: List[HazardBreakdownOut] = []
    debug: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _fetch_flood(lat: float, lon: float) -> Dict[str, Any]:
    """Run the flood prediction pipeline. Returns flood data dict."""
    try:
        from backend.app.ml.flood_service import get_flood_service

        service = get_flood_service()
        logger.debug("[DEBUG] Flood service ready: %s", service.is_ready)

        # If models not loaded, try loading them
        if not service.is_ready:
            from pathlib import Path
            model_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
            logger.info("[DEBUG] Attempting to load models from: %s (exists=%s)", model_dir, model_dir.exists())
            if model_dir.exists():
                service.load_models(model_dir)
                logger.info("[DEBUG] Loaded flood models from %s", model_dir)

        if not service.is_ready:
            logger.warning("Flood models not available — using weather-based heuristic")
            return _flood_heuristic(lat, lon)

        prediction = service.predict(latitude=lat, longitude=lon)
        ens = prediction.ensemble_result

        return {
            "flood_probability": ens.flood_probability,
            "xgb_probability": ens.xgb_probability,
            "lstm_probability": ens.lstm_probability,
            "confidence": ens.confidence,
            "models_agree": ens.models_agree,
            "risk_level": ens.risk_level.value,
            "weather_summary": prediction.weather_summary,
            "source": "ml_ensemble",
        }

    except Exception as e:
        logger.error("Flood prediction failed: %s", e)
        logger.debug(traceback.format_exc())
        return _flood_heuristic(lat, lon)


def _flood_heuristic(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fallback: use raw weather data to estimate flood risk without ML models.
    Uses rainfall accumulation as primary indicator.
    """
    try:
        from backend.app.ingestion.weather_service import fetch_weather

        logger.debug("[DEBUG] Fetching weather for flood heuristic at (%.4f, %.4f)", lat, lon)
        weather = fetch_weather(lat, lon)
        logger.debug(
            "[DEBUG] Weather fetch result: success=%s, status=%s, error=%s",
            weather.success, weather.status.value, weather.error_message
        )
        
        if not weather.success or weather.rainfall is None:
            logger.warning("[DEBUG] Weather fetch failed or no rainfall data")
            return {
                "flood_probability": 0.0,
                "xgb_probability": 0.0,
                "lstm_probability": 0.0,
                "confidence": 0.0,
                "models_agree": True,
                "risk_level": "minimal",
                "weather_summary": {},
                "source": "no_data",
            }

        rain = weather.rainfall
        current = weather.current
        elevation = weather.elevation_m
        
        logger.info(
            "[DEBUG] Rainfall data: 1hr=%.2fmm, 3hr=%.2fmm, 6hr=%.2fmm, 24hr=%.2fmm, elevation=%.1fm",
            rain.rain_1hr, rain.rain_3hr, rain.rain_6hr, rain.rain_24hr, elevation
        )
        
        # Base probability from rainfall
        # 0mm → 0.0, 50mm → 0.3, 100mm → 0.6, 200mm+ → 0.9
        prob = min(0.95, rain.rain_24hr / 220.0)

        # Boost if short-term intensity is high
        if rain.rain_1hr > 20:
            prob = min(0.95, prob + 0.15)
        
        # === Enhanced flood risk factors (even without rain) ===
        humidity = current.humidity_pct if current else 0
        pressure = current.pressure_hpa if current else 1013.25
        soil_moisture = current.soil_moisture if current else 0
        
        # Factor 1: High humidity indicates moisture-laden air (potential rain)
        if humidity >= 85:
            prob = max(prob, 0.15)  # Minimum 15% risk in very humid conditions
        elif humidity >= 75:
            prob = max(prob, 0.08)  # Minimum 8% risk
        elif humidity >= 60:
            prob = max(prob, 0.03)  # Minimum 3% baseline
        
        # Factor 2: Low elevation increases flood susceptibility
        if elevation <= 5:  # Very low-lying area
            prob = min(0.95, prob + 0.10)
        elif elevation <= 15:  # Low-lying area
            prob = min(0.95, prob + 0.05)
        elif elevation <= 30:  # Moderate
            prob = min(0.95, prob + 0.02)
        
        # Factor 3: Low pressure can indicate incoming storm
        if pressure < 1000:
            prob = min(0.95, prob + 0.08)
        elif pressure < 1005:
            prob = min(0.95, prob + 0.04)
        
        # Factor 4: Saturated soil increases runoff risk
        if soil_moisture >= 0.45:
            prob = min(0.95, prob + 0.10)
        elif soil_moisture >= 0.30:
            prob = min(0.95, prob + 0.05)
        
        logger.info(
            "[DEBUG] Flood heuristic factors: humidity=%.0f%%, elevation=%.1fm, pressure=%.1fhPa, soil=%.2f -> prob=%.4f",
            humidity, elevation, pressure, soil_moisture, prob
        )

        # Build weather summary (current already defined above)
        weather_summary = {}
        if current:
            weather_summary = {
                "temperature": current.temperature_c,
                "humidity": current.humidity_pct,
                "wind_speed": current.wind_speed_kmh,
                "pressure": current.pressure_hpa,
                "rain_1hr": rain.rain_1hr,
                "rain_24hr": rain.rain_24hr,
                "elevation": elevation,
                "soil_moisture": soil_moisture,
            }

        return {
            "flood_probability": round(prob, 4),
            "xgb_probability": round(prob, 4),
            "lstm_probability": round(prob, 4),
            "confidence": 0.5,
            "models_agree": True,
            "risk_level": (
                "critical" if prob >= 0.8 else
                "high" if prob >= 0.6 else
                "moderate" if prob >= 0.35 else
                "low" if prob >= 0.15 else "minimal"
            ),
            "weather_summary": weather_summary,
            "source": "weather_heuristic",
        }

    except Exception as e:
        logger.error("Flood heuristic failed: %s", e)
        return {
            "flood_probability": 0.0,
            "xgb_probability": 0.0,
            "lstm_probability": 0.0,
            "confidence": 0.0,
            "models_agree": True,
            "risk_level": "minimal",
            "weather_summary": {},
            "source": "error",
        }


def _fetch_earthquake(lat: float, lon: float, radius_km: float) -> Dict[str, Any]:
    """Fetch earthquake data from USGS + add baseline seismic zone risk."""
    # Get baseline seismic zone risk
    base_risk, zone_name, zone_level = _get_seismic_zone_risk(lat, lon)
    logger.debug(
        "[DEBUG] Seismic zone: %s (Zone %d), baseline_risk=%.2f",
        zone_name, zone_level, base_risk
    )
    
    try:
        from backend.app.ml.earthquake_service import (
            FeedType,
            TimePeriod,
            monitor_earthquakes,
        )

        logger.debug("[DEBUG] Fetching earthquakes: lat=%.4f, lon=%.4f, radius=%.1f km", lat, lon, radius_km)
        
        result = monitor_earthquakes(
            user_lat=lat,
            user_lon=lon,
            radius_km=radius_km,
            min_magnitude=4.0,
            period=TimePeriod.DAY,
            feed_type=FeedType.M4_5_PLUS,  # Fixed: was M4_5 (typo)
        )

        logger.debug(
            "[DEBUG] USGS query: success=%s, total_events=%d",
            result.query_result.success, result.query_result.total_count
        )

        summary = result.summary
        highest_mag = summary.get("highest_magnitude", 0.0) or 0.0
        highest_risk = summary.get("highest_risk", "none")
        events_nearby = summary.get("events_nearby", 0)
        
        logger.info(
            "[DEBUG] Earthquake summary: events_nearby=%d, highest_mag=%.1f, highest_risk=%s",
            events_nearby, highest_mag, highest_risk
        )

        # Also get the depth of the strongest earthquake
        depth_km = 10.0  # default
        if result.filter_result and result.filter_result.matched:
            top_event = max(result.filter_result.matched, key=lambda e: e.magnitude)
            depth_km = top_event.depth_km
            highest_mag = max(highest_mag, top_event.magnitude)
            logger.debug("[DEBUG] Top earthquake: M%.1f at %.1f km depth", top_event.magnitude, depth_km)

        # If no actual earthquakes, use baseline zone risk as pseudo-magnitude
        # Convert baseline risk to a "virtual magnitude" for display
        # baseline 0.25 → ~M4.0 equivalent, 0.15 → ~M3.5, 0.08 → ~M3.0, 0.03 → ~M2.5
        if highest_mag == 0.0 and base_risk > 0:
            # Virtual magnitude based on seismic zone
            virtual_mag = 2.0 + (base_risk * 8.0)  # 0.03 → 2.24, 0.25 → 4.0
            highest_mag = virtual_mag
            highest_risk = "zone_baseline"
            logger.info(
                "[DEBUG] No active earthquakes - using zone baseline: virtual_mag=%.2f",
                virtual_mag
            )

        return {
            "magnitude": highest_mag,
            "depth_km": depth_km,
            "events_nearby": events_nearby,
            "highest_risk": highest_risk,
            "source": "usgs" if events_nearby > 0 else "seismic_zone",
            "seismic_zone": zone_name,
            "zone_level": zone_level,
            "baseline_risk": base_risk,
        }

    except Exception as e:
        logger.error("Earthquake fetch failed: %s", e)
        logger.debug(traceback.format_exc())
        # Even on error, return baseline zone risk
        virtual_mag = 2.0 + (base_risk * 8.0) if base_risk > 0 else 0.0
        return {
            "magnitude": virtual_mag,
            "depth_km": 10.0,
            "events_nearby": 0,
            "highest_risk": "zone_baseline" if virtual_mag > 0 else "none",
            "source": "seismic_zone",
            "seismic_zone": zone_name,
            "zone_level": zone_level,
            "baseline_risk": base_risk,
        }


def _fetch_cyclone(lat: float, lon: float, weather_summary: Dict) -> Dict[str, Any]:
    """Detect cyclone conditions from weather data + add coastal baseline risk."""
    # Get baseline cyclone zone risk
    base_risk, zone_name, is_peak_season = _get_cyclone_zone_risk(lat, lon)
    logger.debug(
        "[DEBUG] Cyclone zone: %s, baseline_risk=%.2f, is_peak_season=%s",
        zone_name, base_risk, is_peak_season
    )
    
    try:
        from backend.app.ml.cyclone_service import (
            assess_cyclone_risk,
            detect_cyclone_conditions,
        )

        # Use weather data if available, otherwise fetch fresh
        wind_speed = weather_summary.get("wind_speed", 0.0) or 0.0
        rainfall = weather_summary.get("rain_24hr", 0.0) or 0.0
        pressure = weather_summary.get("pressure", 1013.25) or 1013.25
        
        logger.debug(
            "[DEBUG] Cyclone input from weather: wind=%.1f km/h, rain=%.1f mm, pressure=%.1f hPa",
            wind_speed, rainfall, pressure
        )

        # If no weather data from flood step, fetch it
        if wind_speed == 0 and rainfall == 0:
            try:
                from backend.app.ingestion.weather_service import fetch_weather
                logger.debug("[DEBUG] Fetching fresh weather data for cyclone detection")
                weather = fetch_weather(lat, lon)
                if weather.success and weather.current:
                    wind_speed = weather.current.wind_speed_kmh
                    pressure = weather.current.pressure_hpa
                if weather.success and weather.rainfall:
                    rainfall = weather.rainfall.rain_24hr
                logger.debug(
                    "[DEBUG] Fresh weather: wind=%.1f km/h, rain=%.1f mm, pressure=%.1f hPa",
                    wind_speed, rainfall, pressure
                )
            except Exception as fetch_err:
                logger.warning("[DEBUG] Fresh weather fetch failed: %s", fetch_err)

        assessment = assess_cyclone_risk(
            latitude=lat,
            longitude=lon,
            wind_speed_kmh=wind_speed,
            rainfall_mm=rainfall,
            pressure_hpa=pressure,
        )

        detection = assessment.detection
        cyclone_score = 0.0
        if detection.risk_score:
            cyclone_score = detection.risk_score.composite_score
        
        # If no active cyclone conditions, use baseline coastal/seasonal risk
        if cyclone_score < 0.05 and base_risk > 0:
            cyclone_score = max(cyclone_score, base_risk)
            logger.info(
                "[DEBUG] No active cyclone - using zone baseline: score=%.4f, zone=%s, peak_season=%s",
                base_risk, zone_name, is_peak_season
            )
        
        logger.info(
            "[DEBUG] Cyclone result: score=%.4f, is_cyclonic=%s, category=%s",
            cyclone_score, detection.is_cyclonic, detection.category.value
        )

        return {
            "cyclone_score": cyclone_score,
            "is_cyclonic": detection.is_cyclonic,
            "category": detection.category.value,
            "escalation": assessment.escalation.value,
            "wind_speed_kmh": wind_speed,
            "conditions_met": detection.conditions_met,
            "source": "weather_analysis" if detection.is_cyclonic else "coastal_zone",
            "cyclone_zone": zone_name,
            "is_peak_season": is_peak_season,
            "baseline_risk": base_risk,
        }

    except Exception as e:
        logger.error("Cyclone detection failed: %s", e)
        logger.debug(traceback.format_exc())
        # Even on error, return baseline coastal risk
        return {
            "cyclone_score": base_risk,
            "is_cyclonic": False,
            "category": "low_pressure",
            "escalation": "none",
            "wind_speed_kmh": 0.0,
            "conditions_met": [],
            "source": "coastal_zone",
            "cyclone_zone": zone_name,
            "is_peak_season": is_peak_season,
            "baseline_risk": base_risk,
        }


def _build_feature_importance(weather_summary: Dict) -> List[Dict]:
    """Build feature importance from weather data for the dashboard."""
    features = []

    rain_24h = weather_summary.get("rain_24hr", 0) or 0
    rain_1h = weather_summary.get("rain_1hr", 0) or 0
    humidity = weather_summary.get("humidity", 0) or 0
    wind = weather_summary.get("wind_speed", 0) or 0
    pressure = weather_summary.get("pressure", 1013) or 1013
    temp = weather_summary.get("temperature", 25) or 25

    # Normalise feature importances (values 0-1)
    features.append({
        "feature": "Rainfall 24h",
        "importance": min(1.0, rain_24h / 100.0),
        "category": "precipitation",
    })
    features.append({
        "feature": "Rainfall 1h",
        "importance": min(1.0, rain_1h / 30.0),
        "category": "precipitation",
    })
    features.append({
        "feature": "Humidity",
        "importance": humidity / 100.0 if humidity else 0,
        "category": "atmosphere",
    })
    features.append({
        "feature": "Wind Speed",
        "importance": min(1.0, wind / 100.0),
        "category": "atmosphere",
    })
    features.append({
        "feature": "Pressure Drop",
        "importance": min(1.0, max(0, (1013.25 - pressure) / 50.0)),
        "category": "atmosphere",
    })
    features.append({
        "feature": "Temperature",
        "importance": min(1.0, max(0, temp / 50.0)),
        "category": "environment",
    })

    # Sort by importance descending
    features.sort(key=lambda f: f["importance"], reverse=True)
    return features


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/assess",
    response_model=AssessResponse,
    summary="Unified Location Risk Assessment",
    description=(
        "End-to-end risk assessment: fetches live weather, runs flood ML models, "
        "queries USGS for earthquakes, detects cyclone conditions, and aggregates "
        "everything into a unified risk score."
    ),
)
async def assess_location_risk(req: AssessRequest):
    """
    Full pipeline risk assessment for a location.

    1. Fetch weather → predict flood probability
    2. Fetch USGS earthquake data
    3. Detect cyclone conditions
    4. Aggregate into unified risk score
    """
    lat, lon, radius = req.latitude, req.longitude, req.radius_km

    logger.info(
        "=== RISK ASSESSMENT: lat=%.4f, lon=%.4f, radius=%.0f km ===",
        lat, lon, radius,
    )

    # ── Step 1: Flood prediction ──
    flood = _fetch_flood(lat, lon)
    flood_prob = flood["flood_probability"]
    logger.info(
        "FLOOD: prob=%.4f, source=%s, rain_24h=%s",
        flood_prob,
        flood.get("source"),
        flood.get("weather_summary", {}).get("rain_24hr"),
    )

    # ── Step 2: Earthquake monitoring ──
    earthquake = _fetch_earthquake(lat, lon, radius)
    eq_magnitude = earthquake["magnitude"]
    eq_depth = earthquake["depth_km"]
    logger.info(
        "EARTHQUAKE: mag=%.1f, depth=%.0f km, events=%d, source=%s",
        eq_magnitude, eq_depth,
        earthquake.get("events_nearby", 0),
        earthquake.get("source"),
    )

    # ── Step 3: Cyclone detection ──
    cyclone = _fetch_cyclone(lat, lon, flood.get("weather_summary", {}))
    cyclone_score = cyclone["cyclone_score"]
    logger.info(
        "CYCLONE: score=%.4f, is_cyclonic=%s, category=%s, wind=%.1f km/h",
        cyclone_score,
        cyclone.get("is_cyclonic"),
        cyclone.get("category"),
        cyclone.get("wind_speed_kmh", 0),
    )

    # ── Step 4: Risk aggregation ──
    try:
        from backend.app.ml.risk_aggregator import (
            ALERT_MESSAGES,
            aggregate_risk,
        )

        agg = aggregate_risk(
            flood_probability=flood_prob,
            earthquake_magnitude=eq_magnitude,
            earthquake_depth_km=eq_depth,
            cyclone_score=cyclone_score,
            latitude=lat,
            longitude=lon,
        )

        agg_dict = agg.to_dict()
        logger.info(
            "AGGREGATED: score=%.1f%%, level=%s, dominant=%s, active=%d",
            agg.overall_risk_score,
            agg.overall_risk_level.value,
            agg.dominant_hazard,
            agg.active_hazard_count,
        )

    except Exception as e:
        logger.error("Risk aggregation failed: %s", e)
        logger.debug(traceback.format_exc())
        # Fallback: simple max
        max_score = max(flood_prob * 100, eq_magnitude * 10, cyclone_score * 100)
        agg_dict = {
            "overall_risk_score": max_score,
            "overall_risk_level": (
                "severe" if max_score >= 60 else
                "warning" if max_score >= 40 else
                "watch" if max_score >= 15 else "safe"
            ),
            "dominant_hazard": "flood",
            "active_hazard_count": sum([
                flood_prob > 0.1,
                eq_magnitude >= 4.0,
                cyclone_score > 0.15,
            ]),
            "alert_triggered": max_score >= 15,
            "alert_reasons": [],
            "hazard_breakdown": [],
        }

    # ── Build model confidence ──
    model_confidence = ModelConfidenceOut(
        xgboost=flood.get("xgb_probability", 0.0),
        lstm=flood.get("lstm_probability", 0.0),
        ensemble=flood_prob,
        agreement=flood.get("models_agree", True),
    )

    # ── Build feature importance ──
    feature_importance = _build_feature_importance(
        flood.get("weather_summary", {})
    )

    # ── Build hazard breakdown ──
    hazard_breakdown = agg_dict.get("hazard_breakdown", [])

    # ── Build alert reasons ──
    alert_reasons = agg_dict.get("alert_reasons", [])

    # Add contextual reasons
    if flood_prob > 0.1:
        rain_info = flood.get("weather_summary", {}).get("rain_24hr")
        if rain_info:
            alert_reasons.append(f"Rainfall: {rain_info:.1f} mm in 24h")
    if eq_magnitude >= 4.0:
        alert_reasons.append(
            f"Earthquake M{eq_magnitude:.1f} detected within {radius:.0f} km"
        )
    if cyclone.get("is_cyclonic"):
        alert_reasons.append(
            f"Cyclonic conditions: {cyclone.get('category', 'unknown')} "
            f"(wind {cyclone.get('wind_speed_kmh', 0):.0f} km/h)"
        )

    # ── Build response ──
    response = AssessResponse(
        overall_risk_score=agg_dict.get("overall_risk_score", 0),
        overall_risk_level=agg_dict.get("overall_risk_level", "safe"),
        dominant_hazard=agg_dict.get("dominant_hazard", "none"),
        alert_action=agg_dict.get("alert_action", "No action required"),
        alert_reasons=alert_reasons,
        active_hazard_count=agg_dict.get("active_hazard_count", 0),
        ensemble_alpha=0.65,
        model_confidence=model_confidence,
        feature_importance=[
            FeatureImportanceOut(**f) for f in feature_importance
        ],
        hazard_breakdown=[
            HazardBreakdownOut(**h) if isinstance(h, dict) else h
            for h in hazard_breakdown
        ],
        debug={
            "flood_source": flood.get("source"),
            "earthquake_source": earthquake.get("source"),
            "cyclone_source": cyclone.get("source"),
            "flood_prob": flood_prob,
            "earthquake_mag": eq_magnitude,
            "cyclone_score": cyclone_score,
            "weather": flood.get("weather_summary", {}),
        },
    )

    logger.info("=== ASSESSMENT COMPLETE ===")
    return response


# ---------------------------------------------------------------------------
# Debug & Simulation Endpoints
# ---------------------------------------------------------------------------

class DebugWeatherResponse(BaseModel):
    """Debug response showing raw weather API data."""
    success: bool
    api_url: str
    latitude: float
    longitude: float
    elevation_m: float
    current_temperature: float
    current_humidity: float
    current_wind_speed: float
    current_pressure: float
    rain_1hr: float
    rain_3hr: float
    rain_6hr: float
    rain_24hr: float
    raw_precipitation_24h: List[float]
    flood_probability: float
    flood_risk_category: str
    source: str
    error: Optional[str] = None


@router.get(
    "/debug/weather",
    response_model=DebugWeatherResponse,
    summary="Debug Weather API",
    description="Test the Open-Meteo API call and see raw data for debugging.",
)
async def debug_weather(
    lat: float = 12.9245,
    lon: float = 80.0880,
):
    """
    Debug endpoint to verify Open-Meteo API integration.
    
    Shows:
    - Raw API response data
    - Rainfall accumulation calculations
    - Final flood probability
    """
    from backend.app.ingestion.weather_service import fetch_weather, OPEN_METEO_BASE_URL
    
    api_url = f"{OPEN_METEO_BASE_URL}?latitude={lat}&longitude={lon}&hourly=precipitation,rain&past_days=1"
    
    weather = fetch_weather(lat, lon)
    
    if not weather.success:
        return DebugWeatherResponse(
            success=False,
            api_url=api_url,
            latitude=lat,
            longitude=lon,
            elevation_m=0,
            current_temperature=0,
            current_humidity=0,
            current_wind_speed=0,
            current_pressure=0,
            rain_1hr=0,
            rain_3hr=0,
            rain_6hr=0,
            rain_24hr=0,
            raw_precipitation_24h=[],
            flood_probability=0,
            flood_risk_category="error",
            source="error",
            error=weather.error_message,
        )
    
    # Get raw precipitation values (last 24 hours)
    raw_precip = []
    if weather.hourly and weather.hourly.precipitation:
        raw_precip = weather.hourly.precipitation[-24:]
    
    # Calculate flood probability using the heuristic
    flood_data = _flood_heuristic(lat, lon)
    
    return DebugWeatherResponse(
        success=True,
        api_url=api_url,
        latitude=weather.latitude,
        longitude=weather.longitude,
        elevation_m=weather.elevation_m,
        current_temperature=weather.current.temperature_c if weather.current else 0,
        current_humidity=weather.current.humidity_pct if weather.current else 0,
        current_wind_speed=weather.current.wind_speed_kmh if weather.current else 0,
        current_pressure=weather.current.pressure_hpa if weather.current else 0,
        rain_1hr=weather.rainfall.rain_1hr if weather.rainfall else 0,
        rain_3hr=weather.rainfall.rain_3hr if weather.rainfall else 0,
        rain_6hr=weather.rainfall.rain_6hr if weather.rainfall else 0,
        rain_24hr=weather.rainfall.rain_24hr if weather.rainfall else 0,
        raw_precipitation_24h=raw_precip,
        flood_probability=flood_data.get("flood_probability", 0),
        flood_risk_category=flood_data.get("risk_level", "unknown"),
        source=flood_data.get("source", "unknown"),
    )


class SimulateRequest(BaseModel):
    """Request to simulate disaster conditions for testing."""
    latitude: float = Field(default=12.9245)
    longitude: float = Field(default=80.0880)
    radius_km: float = Field(default=50.0)
    # Simulated values (override real API data)
    simulated_rain_24hr: float = Field(default=0.0, description="Simulated 24hr rainfall in mm")
    simulated_earthquake_magnitude: float = Field(default=0.0, description="Simulated earthquake magnitude")
    simulated_earthquake_depth_km: float = Field(default=10.0, description="Simulated earthquake depth")
    simulated_wind_speed_kmh: float = Field(default=0.0, description="Simulated wind speed for cyclone")
    simulated_pressure_hpa: float = Field(default=1013.0, description="Simulated pressure for cyclone")


@router.post(
    "/simulate",
    response_model=AssessResponse,
    summary="Simulate Disaster Conditions",
    description="Test the dashboard with simulated disaster values (rainfall, earthquake, cyclone).",
)
async def simulate_disaster(req: SimulateRequest):
    """
    Simulate disaster conditions for testing.
    
    Use this to test the dashboard with various risk levels:
    - Set simulated_rain_24hr=100 for HIGH flood risk
    - Set simulated_earthquake_magnitude=6.5 for HIGH earthquake risk
    - Set simulated_wind_speed_kmh=120 for CYCLONE conditions
    """
    lat, lon, radius = req.latitude, req.longitude, req.radius_km
    
    logger.info(
        "=== SIMULATION MODE: lat=%.4f, lon=%.4f, rain=%.1fmm, eq=M%.1f, wind=%.1fkm/h ===",
        lat, lon, req.simulated_rain_24hr, req.simulated_earthquake_magnitude, req.simulated_wind_speed_kmh
    )
    
    # ── Step 1: Simulated flood ──
    rain_24hr = req.simulated_rain_24hr
    flood_prob = min(0.95, rain_24hr / 220.0)
    if rain_24hr > 100:
        flood_prob = min(0.95, flood_prob + 0.15)
    
    flood = {
        "flood_probability": round(flood_prob, 4),
        "xgb_probability": round(flood_prob, 4),
        "lstm_probability": round(flood_prob, 4),
        "confidence": 0.8,
        "models_agree": True,
        "risk_level": (
            "critical" if flood_prob >= 0.8 else
            "high" if flood_prob >= 0.6 else
            "moderate" if flood_prob >= 0.35 else
            "low" if flood_prob >= 0.15 else "minimal"
        ),
        "weather_summary": {
            "temperature": 28.0,
            "humidity": 85.0,
            "wind_speed": req.simulated_wind_speed_kmh,
            "pressure": req.simulated_pressure_hpa,
            "rain_1hr": rain_24hr / 10,  # Approximate
            "rain_24hr": rain_24hr,
        },
        "source": "simulation",
    }
    
    # ── Step 2: Simulated earthquake ──
    earthquake = {
        "magnitude": req.simulated_earthquake_magnitude,
        "depth_km": req.simulated_earthquake_depth_km,
        "events_nearby": 1 if req.simulated_earthquake_magnitude > 0 else 0,
        "highest_risk": "high" if req.simulated_earthquake_magnitude >= 6.0 else "moderate" if req.simulated_earthquake_magnitude >= 4.5 else "low",
        "source": "simulation",
    }
    
    # ── Step 3: Simulated cyclone ──
    wind_speed = req.simulated_wind_speed_kmh
    pressure = req.simulated_pressure_hpa
    
    # Cyclone score based on wind and pressure
    wind_factor = min(1.0, wind_speed / 180.0)  # 180 km/h = max
    pressure_factor = min(1.0, max(0, (1013 - pressure) / 80.0))  # Low pressure contributes
    cyclone_score = (wind_factor * 0.6 + pressure_factor * 0.4)
    
    is_cyclonic = wind_speed >= 62 or pressure < 1000
    
    cyclone = {
        "cyclone_score": round(cyclone_score, 4),
        "is_cyclonic": is_cyclonic,
        "category": "cyclone" if wind_speed >= 119 else "severe_storm" if wind_speed >= 89 else "storm" if wind_speed >= 62 else "low_pressure",
        "escalation": "critical" if wind_speed >= 119 else "high" if wind_speed >= 89 else "moderate" if wind_speed >= 62 else "none",
        "wind_speed_kmh": wind_speed,
        "conditions_met": [],
        "source": "simulation",
    }
    
    # ── Step 4: Risk aggregation ──
    from backend.app.ml.risk_aggregator import aggregate_risk, ALERT_MESSAGES
    
    agg = aggregate_risk(
        flood_probability=flood_prob,
        earthquake_magnitude=req.simulated_earthquake_magnitude,
        earthquake_depth_km=req.simulated_earthquake_depth_km,
        cyclone_score=cyclone_score,
        latitude=lat,
        longitude=lon,
    )
    
    agg_dict = agg.to_dict()
    
    # Build response (same structure as real assess endpoint)
    model_confidence = ModelConfidenceOut(
        xgboost=flood.get("xgb_probability", 0.0),
        lstm=flood.get("lstm_probability", 0.0),
        ensemble=flood_prob,
        agreement=True,
    )
    
    feature_importance = _build_feature_importance(flood.get("weather_summary", {}))
    hazard_breakdown = agg_dict.get("hazard_breakdown", [])
    alert_reasons = agg_dict.get("alert_reasons", [])
    
    # Add simulation context
    if rain_24hr > 0:
        alert_reasons.append(f"[SIMULATED] Rainfall: {rain_24hr:.1f} mm in 24h")
    if req.simulated_earthquake_magnitude > 0:
        alert_reasons.append(f"[SIMULATED] Earthquake M{req.simulated_earthquake_magnitude:.1f}")
    if wind_speed >= 62:
        alert_reasons.append(f"[SIMULATED] Cyclonic winds: {wind_speed:.0f} km/h")
    
    return AssessResponse(
        overall_risk_score=agg_dict.get("overall_risk_score", 0),
        overall_risk_level=agg_dict.get("overall_risk_level", "safe"),
        dominant_hazard=agg_dict.get("dominant_hazard", "none"),
        alert_action=agg_dict.get("alert_action", "No action required"),
        alert_reasons=alert_reasons,
        active_hazard_count=agg_dict.get("active_hazard_count", 0),
        ensemble_alpha=0.65,
        model_confidence=model_confidence,
        feature_importance=[FeatureImportanceOut(**f) for f in feature_importance],
        hazard_breakdown=[
            HazardBreakdownOut(**h) if isinstance(h, dict) else h
            for h in hazard_breakdown
        ],
        debug={
            "mode": "SIMULATION",
            "flood_source": "simulation",
            "earthquake_source": "simulation",
            "cyclone_source": "simulation",
            "flood_prob": flood_prob,
            "earthquake_mag": req.simulated_earthquake_magnitude,
            "cyclone_score": cyclone_score,
            "weather": flood.get("weather_summary", {}),
        },
    )
