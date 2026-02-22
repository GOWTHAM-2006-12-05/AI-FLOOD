/**
 * API client — typed interface to the FastAPI backend.
 *
 * All fetch calls go through a single `request()` wrapper that handles
 * base URL, error normalisation, and JSON parsing.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ──

export interface RiskData {
  overall_risk_score: number;
  overall_risk_level: "safe" | "watch" | "warning" | "severe";
  dominant_hazard: string;
  alert_action: string;
  alert_reasons: string[];
  active_hazard_count: number;
  ensemble_alpha?: number;
  model_confidence?: ModelConfidence;
  feature_importance?: FeatureImportance[];
  hazard_breakdown?: HazardBreakdown[];
}

export interface HazardBreakdown {
  hazard_type: string;
  raw_value: number;
  normalised_score: number;
  weight: number;
  weighted_contribution: number;
  is_active: boolean;
  is_critical: boolean;
  priority: number;
}

export interface ModelConfidence {
  xgboost: number;
  lstm: number;
  ensemble: number;
  agreement: boolean;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  category: string;
}

export interface ForecastHorizon {
  minutes: number;
  label: string;
}

export interface GridCell {
  row: number;
  col: number;
  lat: number;
  lng: number;
  flood_risk: number;
  elevation: number;
  drainage: number;
}

export interface WeatherData {
  temperature: number;
  humidity: number;
  wind_speed: number;
  rainfall_1h: number;
  rainfall_3h: number;
  rainfall_6h: number;
  rainfall_24h: number;
  pressure: number;
}

export interface AlertBroadcast {
  alert_id: string;
  priority: string;
  total_recipients: number;
  recipients_reached: number;
  reach_rate: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  environment: string;
  components: { name: string; status: string; latency_ms: number }[];
}

// ── Request helper ──

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${path}`;

  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(
      error?.error?.message || `API error ${res.status}: ${res.statusText}`
    );
  }

  return res.json();
}

// ── API Client ──

export const apiClient = {
  // ── Health ──
  getHealth: () => request<HealthStatus>("/health"),

  // ── Weather ──
  getWeather: (lat: number, lon: number) =>
    request<WeatherData>(`/api/v1/weather/current?lat=${lat}&lon=${lon}`),

  // ── Flood ──
  getFloodPrediction: (lat: number, lon: number) =>
    request<any>("/api/v1/flood/predict", {
      method: "POST",
      body: JSON.stringify({ latitude: lat, longitude: lon }),
    }),

  // ── Risk Aggregation ──
  // Uses the full pipeline endpoint that fetches live weather, earthquakes, & cyclone data
  getRiskAssessment: (lat: number, lon: number, radiusKm: number = 50) =>
    request<RiskData>("/api/v1/risk/assess", {
      method: "POST",
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        radius_km: radiusKm,
      }),
    }),

  // ── Grid Simulation ──
  getGridSimulation: (lat: number, lon: number, radiusKm: number = 10) =>
    request<{ cells: GridCell[] }>("/api/v1/grid/simulate", {
      method: "POST",
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        radius_km: radiusKm,
      }),
    }),

  // ── Forecast ──
  getForecast: (lat: number, lon: number, horizonMinutes: number = 30) =>
    request<any>("/api/v1/forecast/predict", {
      method: "POST",
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        horizon_minutes: horizonMinutes,
      }),
    }),

  // ── Earthquake ──
  getEarthquakes: (lat: number, lon: number, radiusKm: number = 200) =>
    request<any>(
      `/api/v1/earthquake/nearby?lat=${lat}&lon=${lon}&radius_km=${radiusKm}`
    ),

  // ── Cyclone ──
  getCyclones: (lat: number, lon: number) =>
    request<any>(
      `/api/v1/cyclone/active?lat=${lat}&lon=${lon}`
    ),

  // ── Alerts ──
  broadcastAlert: (data: any) =>
    request<AlertBroadcast>("/api/v1/alerts/broadcast/risk", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  getAlertStatus: (alertId: string) =>
    request<any>(`/api/v1/alerts/${alertId}/status`),

  // ── Disasters (nearby) ──
  getNearbyDisasters: (
    lat: number,
    lon: number,
    radiusKm: number = 50
  ) =>
    request<any>("/api/v1/disasters/nearby", {
      method: "POST",
      body: JSON.stringify({
        location: { latitude: lat, longitude: lon, source: "manual" },
        radius_km: radiusKm,
      }),
    }),
};
