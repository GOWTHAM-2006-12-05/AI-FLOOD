"""
forecast_engine.py — Sub-hour flood forecasting engine.

Provides multi-horizon rainfall and flood-probability forecasts at:
    • T+30 min  (nowcasting)
    • T+1 hr    (short-range)
    • T+3 hr    (medium-range)

Core capabilities:
    1. Time-series input window construction from raw sensor / API data.
    2. Rainfall burst detection (gradient + threshold analysis).
    3. Rainfall acceleration trend estimation (second-order derivative).
    4. Rolling prediction with sliding window advancement.
    5. Per-horizon confidence interval estimation via quantile regression.
    6. Ensemble combination with XGBoost for final flood probability.

Designed for real-time operation:
    - Each prediction cycle ingests the latest readings, slides the window
      forward, and emits forecasts for all three horizons simultaneously.
    - Stateless: all temporal state is carried in the input window.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Raw observations (10-min interval, last 3 hours)    │
    │  → 18 timesteps × n_features input tensor            │
    └──────────────┬───────────────────────────────────────┘
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  BURST DETECTOR            TREND ESTIMATOR           │
    │  (gradient threshold)      (2nd-order regression)    │
    └──────────────┬───────────────────────────────────────┘
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  LSTM Forecast Head  ──→  3 outputs per horizon      │
    │   P_lower (10th %ile)                                │
    │   P_median (50th %ile)                               │
    │   P_upper (90th %ile)                                │
    └──────────────┬───────────────────────────────────────┘
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  ENSEMBLE COMBINER                                   │
    │   XGBoost snapshot features  ←─  latest tabular row  │
    │   α·XGB + (1-α)·LSTM_median  = P_flood              │
    └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================
#  CONSTANTS
# ===================================================================

# Observation interval in minutes
OBS_INTERVAL_MIN = 10

# Window design: 3 hours of 10-min observations = 18 timesteps
WINDOW_TIMESTEPS = 18
WINDOW_DURATION_MIN = WINDOW_TIMESTEPS * OBS_INTERVAL_MIN  # 180 min

# Forecast horizons (minutes ahead)
class Horizon(int, Enum):
    """Forecast horizon identifiers."""
    MIN_30 = 30
    HOUR_1 = 60
    HOUR_3 = 180

ALL_HORIZONS: List[Horizon] = [Horizon.MIN_30, Horizon.HOUR_1, Horizon.HOUR_3]

# Horizon labels for display
HORIZON_LABELS: Dict[Horizon, str] = {
    Horizon.MIN_30: "T+30 min",
    Horizon.HOUR_1: "T+1 hr",
    Horizon.HOUR_3: "T+3 hr",
}

# Features expected per timestep
FORECAST_FEATURES: List[str] = [
    "rainfall_mm",           # instantaneous rainfall over interval
    "rainfall_cumulative",   # running total since window start
    "rainfall_rate_mm_hr",   # mm/hr equivalent
    "soil_moisture",         # volumetric water content 0-1
    "temperature_c",         # temperature in Celsius
    "relative_humidity",     # percentage
    "surface_pressure_hpa",  # hPa
    "wind_speed_ms",         # m/s
]

N_FEATURES = len(FORECAST_FEATURES)

# Burst detection thresholds
BURST_RATE_THRESHOLD_MM_HR = 20.0      # rainfall rate to flag burst
BURST_GRADIENT_THRESHOLD = 5.0          # mm/hr increase per step
BURST_SUSTAINED_STEPS = 2               # min consecutive steps above threshold

# Confidence quantiles
QUANTILE_LOWER = 0.10
QUANTILE_UPPER = 0.90


# ===================================================================
#  DATA STRUCTURES
# ===================================================================


@dataclass
class Observation:
    """Single 10-minute observation from sensors / API."""
    timestamp_unix: float           # epoch seconds
    rainfall_mm: float = 0.0       # rain in this 10-min interval
    soil_moisture: float = 0.3
    temperature_c: float = 28.0
    relative_humidity: float = 70.0
    surface_pressure_hpa: float = 1013.0
    wind_speed_ms: float = 3.0

    def to_feature_vector(self, cumulative_rain: float = 0.0) -> np.ndarray:
        """Convert to the 8-feature vector expected by the model."""
        rate_mm_hr = self.rainfall_mm * (60.0 / OBS_INTERVAL_MIN)
        return np.array([
            self.rainfall_mm,
            cumulative_rain,
            rate_mm_hr,
            self.soil_moisture,
            self.temperature_c,
            self.relative_humidity,
            self.surface_pressure_hpa,
            self.wind_speed_ms,
        ], dtype=np.float64)


@dataclass
class BurstDetection:
    """Result of rainfall burst analysis."""
    is_burst: bool = False
    burst_intensity_mm_hr: float = 0.0
    burst_duration_steps: int = 0
    burst_start_index: int = -1
    max_rate_mm_hr: float = 0.0
    mean_rate_mm_hr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_burst": self.is_burst,
            "burst_intensity_mm_hr": round(self.burst_intensity_mm_hr, 2),
            "burst_duration_steps": self.burst_duration_steps,
            "burst_duration_minutes": self.burst_duration_steps * OBS_INTERVAL_MIN,
            "max_rate_mm_hr": round(self.max_rate_mm_hr, 2),
            "mean_rate_mm_hr": round(self.mean_rate_mm_hr, 2),
        }


@dataclass
class TrendEstimate:
    """Rainfall acceleration / deceleration trend."""
    slope: float = 0.0              # mm/hr per step (1st derivative)
    acceleration: float = 0.0       # mm/hr² per step (2nd derivative)
    trend_direction: str = "stable"  # "intensifying", "stable", "weakening"
    r_squared: float = 0.0          # goodness of fit
    projected_rate_mm_hr: float = 0.0  # extrapolated rate at next step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope_mm_hr_per_step": round(self.slope, 4),
            "acceleration": round(self.acceleration, 4),
            "trend_direction": self.trend_direction,
            "r_squared": round(self.r_squared, 4),
            "projected_rate_mm_hr": round(self.projected_rate_mm_hr, 2),
        }


@dataclass
class HorizonForecast:
    """Forecast for a single horizon."""
    horizon: Horizon
    horizon_label: str
    flood_probability: float          # point estimate (median)
    confidence_lower: float           # 10th percentile
    confidence_upper: float           # 90th percentile
    confidence_width: float           # upper - lower
    rainfall_forecast_mm: float       # predicted accumulation to horizon
    risk_level: str                   # minimal / low / moderate / high / critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_minutes": self.horizon.value,
            "horizon_label": self.horizon_label,
            "flood_probability": round(self.flood_probability, 4),
            "confidence_interval": {
                "lower": round(self.confidence_lower, 4),
                "upper": round(self.confidence_upper, 4),
                "width": round(self.confidence_width, 4),
            },
            "rainfall_forecast_mm": round(self.rainfall_forecast_mm, 2),
            "risk_level": self.risk_level,
        }


@dataclass
class ForecastResult:
    """Complete multi-horizon forecast output."""
    forecasts: List[HorizonForecast] = field(default_factory=list)
    burst_detection: Optional[BurstDetection] = None
    trend_estimate: Optional[TrendEstimate] = None
    ensemble_details: Dict[str, Any] = field(default_factory=dict)
    window_stats: Dict[str, Any] = field(default_factory=dict)
    model_type: str = "fallback"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecasts": [f.to_dict() for f in self.forecasts],
            "burst_detection": self.burst_detection.to_dict() if self.burst_detection else None,
            "trend_estimate": self.trend_estimate.to_dict() if self.trend_estimate else None,
            "ensemble_details": self.ensemble_details,
            "window_stats": self.window_stats,
            "model_type": self.model_type,
        }


# ===================================================================
#  1. INPUT WINDOW CONSTRUCTION
# ===================================================================


def build_input_window(
    observations: List[Observation],
    window_size: int = WINDOW_TIMESTEPS,
) -> np.ndarray:
    """
    Construct the 2-D input matrix from raw observations.

    The window uses the most recent `window_size` observations.
    If fewer are available, left-pads with zeros (cold-start).

    Input window design rationale:
    ─────────────────────────────
    • 10-minute interval captures sub-hour dynamics that hourly data misses.
    • 18-step window (3 hours) provides enough context for 3-hour forecasts.
    • Cumulative rainfall feature adds monotonic trend information.
    • Rate (mm/hr) normalises the variable-interval raw readings.

    Parameters
    ----------
    observations : list of Observation
        Chronologically ordered (oldest first).
    window_size : int
        Number of timesteps in the window.

    Returns
    -------
    np.ndarray, shape (window_size, N_FEATURES)
        Ready for model ingestion (after optional normalisation).
    """
    n_obs = len(observations)
    matrix = np.zeros((window_size, N_FEATURES), dtype=np.float64)

    # Use the tail of the observations list
    start = max(0, n_obs - window_size)
    relevant = observations[start:]

    # Offset for left-padding when fewer observations than window
    offset = window_size - len(relevant)

    cumulative_rain = 0.0
    for i, obs in enumerate(relevant):
        cumulative_rain += obs.rainfall_mm
        matrix[offset + i] = obs.to_feature_vector(cumulative_rain)

    return matrix


def normalise_window(
    window: np.ndarray,
    feature_means: Optional[np.ndarray] = None,
    feature_stds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise the input window.

    If means/stds are not provided, computes them from the window itself
    (suitable for inference when no pre-fitted scaler is available).

    Returns (normalised_window, means, stds).
    """
    if feature_means is None:
        feature_means = window.mean(axis=0)
    if feature_stds is None:
        feature_stds = window.std(axis=0) + 1e-8

    normalised = (window - feature_means) / feature_stds
    return normalised, feature_means, feature_stds


# ===================================================================
#  2. RAINFALL BURST DETECTION
# ===================================================================


def detect_burst(
    window: np.ndarray,
    rate_threshold: float = BURST_RATE_THRESHOLD_MM_HR,
    gradient_threshold: float = BURST_GRADIENT_THRESHOLD,
    sustained_steps: int = BURST_SUSTAINED_STEPS,
) -> BurstDetection:
    """
    Detect rainfall bursts in the observation window.

    A burst is detected when:
        1. Rainfall rate exceeds `rate_threshold` mm/hr, AND
        2. The rate has been above threshold for ≥ `sustained_steps`, OR
        3. The rate gradient (step-over-step increase) exceeds
           `gradient_threshold` mm/hr per step.

    Burst detection strategy:
    ─────────────────────────
    • Extract the rainfall_rate_mm_hr column (index 2).
    • Compute forward differences (gradient).
    • Flag steps where rate > threshold.
    • Find the longest consecutive run above threshold.
    • Also flag single-step spikes where gradient is extreme.

    Parameters
    ----------
    window : np.ndarray, shape (T, N_FEATURES)
        The un-normalised input window.
    rate_threshold : float
        mm/hr above which rainfall is considered intense.
    gradient_threshold : float
        mm/hr increase per step to flag acceleration.
    sustained_steps : int
        Minimum consecutive steps above threshold for a burst.

    Returns
    -------
    BurstDetection
    """
    rain_rate = window[:, 2]  # rainfall_rate_mm_hr column
    n = len(rain_rate)

    if n == 0 or rain_rate.max() == 0:
        return BurstDetection()

    # Compute step-over-step gradient
    gradients = np.diff(rain_rate)  # length n-1

    # Find consecutive runs above threshold
    above = rain_rate >= rate_threshold
    max_run = 0
    current_run = 0
    burst_start = -1
    best_start = -1

    for i in range(n):
        if above[i]:
            if current_run == 0:
                burst_start = i
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                best_start = burst_start
        else:
            current_run = 0

    # Check gradient spike
    has_gradient_spike = bool(
        len(gradients) > 0 and np.any(gradients >= gradient_threshold)
    )

    # A burst is either a sustained period or a gradient spike
    is_burst = (max_run >= sustained_steps) or has_gradient_spike

    # Burst intensity = max rate during the burst period
    if best_start >= 0:
        burst_end = best_start + max_run
        burst_intensity = float(rain_rate[best_start:burst_end].max())
    else:
        burst_intensity = float(rain_rate.max())

    return BurstDetection(
        is_burst=is_burst,
        burst_intensity_mm_hr=burst_intensity,
        burst_duration_steps=max_run,
        burst_start_index=best_start,
        max_rate_mm_hr=float(rain_rate.max()),
        mean_rate_mm_hr=float(rain_rate.mean()),
    )


# ===================================================================
#  3. RAINFALL ACCELERATION TREND
# ===================================================================


def estimate_trend(
    window: np.ndarray,
    lookback_steps: int = 6,
) -> TrendEstimate:
    """
    Estimate rainfall rate trend via polynomial regression.

    Fits a quadratic to the last `lookback_steps` of rainfall rate:
        rate(t) = a·t² + b·t + c

    Where:
        • b = slope (1st derivative at last point) → mm/hr per step
        • 2a = acceleration (2nd derivative) → mm/hr² per step

    Trend classification:
        • acceleration > +1.0  → "intensifying" (rain getting heavier faster)
        • acceleration < -1.0  → "weakening"    (rain easing off)
        • otherwise            → "stable"

    Why quadratic?
    ──────────────
    Linear regression captures steady intensification but misses
    acceleration. Quadratic captures whether the rain is getting
    heavier at an *increasing* rate — critical for flash floods where
    the intensity curve is concave up.

    Parameters
    ----------
    window : np.ndarray, shape (T, N_FEATURES)
        The un-normalised input window.
    lookback_steps : int
        Number of recent steps to use for trend fitting.

    Returns
    -------
    TrendEstimate
    """
    rain_rate = window[:, 2]  # rainfall_rate_mm_hr
    n = len(rain_rate)

    # Use the last `lookback_steps` points
    tail = rain_rate[-min(lookback_steps, n):]
    m = len(tail)

    if m < 3:
        return TrendEstimate(
            projected_rate_mm_hr=float(tail[-1]) if m > 0 else 0.0,
        )

    t = np.arange(m, dtype=np.float64)

    # Fit quadratic: rate = a·t² + b·t + c
    coeffs = np.polyfit(t, tail, deg=2)
    a, b, c = coeffs

    # Evaluate at last point
    t_last = float(m - 1)
    slope = 2 * a * t_last + b         # dr/dt at last point
    acceleration = 2 * a               # d²r/dt²

    # Extrapolate one step ahead
    t_next = t_last + 1
    projected = a * t_next**2 + b * t_next + c
    projected = max(projected, 0.0)  # rain can't be negative

    # R² for goodness of fit
    fitted = np.polyval(coeffs, t)
    ss_res = np.sum((tail - fitted) ** 2)
    ss_tot = np.sum((tail - tail.mean()) ** 2) + 1e-10
    r_squared = float(max(1.0 - ss_res / ss_tot, 0.0))

    # Classify direction
    if acceleration > 1.0:
        direction = "intensifying"
    elif acceleration < -1.0:
        direction = "weakening"
    else:
        direction = "stable"

    return TrendEstimate(
        slope=float(slope),
        acceleration=float(acceleration),
        trend_direction=direction,
        r_squared=r_squared,
        projected_rate_mm_hr=float(projected),
    )


# ===================================================================
#  4. LSTM FORECAST MODEL (MULTI-HORIZON)
# ===================================================================


class _FallbackForecaster:
    """
    Lightweight numpy-only forecaster used when TensorFlow is unavailable.

    Uses exponential smoothing + physics-based heuristics:
        • Extrapolates recent rainfall trend.
        • Applies empirical flood probability curve.
        • Generates synthetic confidence intervals from variance.

    NOT production quality — placeholder for demo/testing.
    """

    def __init__(self, alpha: float = 0.3, seed: int = 42):
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        self._fitted = False
        self._rain_mean = 0.0
        self._rain_std = 1.0
        self._soil_mean = 0.3

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Fit on historical data. X: (n_samples, seq_len, n_features)."""
        # Learn distribution of rainfall
        rain_col = X[:, :, 0]  # rainfall_mm
        self._rain_mean = float(rain_col.mean())
        self._rain_std = float(rain_col.std() + 1e-8)
        self._soil_mean = float(X[:, :, 3].mean())  # soil_moisture
        self._fitted = True
        return {"rain_mean": self._rain_mean, "rain_std": self._rain_std}

    def predict_horizons(
        self,
        window: np.ndarray,
    ) -> Dict[Horizon, Tuple[float, float, float, float]]:
        """
        Predict flood probability for each horizon.

        Returns dict mapping Horizon → (p_lower, p_median, p_upper, rain_forecast_mm).
        """
        rain_rate = window[:, 2]  # rainfall_rate_mm_hr
        soil = window[:, 3]      # soil_moisture

        # Current state
        current_rate = float(rain_rate[-1]) if len(rain_rate) > 0 else 0.0
        current_soil = float(soil[-1]) if len(soil) > 0 else 0.3

        # Exponential smoothing for rate extrapolation
        smoothed_rate = current_rate
        for r in rain_rate[-6:]:
            smoothed_rate = self.alpha * r + (1 - self.alpha) * smoothed_rate

        results = {}
        for horizon in ALL_HORIZONS:
            hours = horizon.value / 60.0

            # Extrapolate rainfall accumulation
            rain_forecast = smoothed_rate * hours
            # Add uncertainty that grows with horizon
            uncertainty = 0.1 * hours * max(smoothed_rate, 1.0)

            # Flood probability: logistic based on rate, soil, and accumulation
            x = (
                0.4 * (smoothed_rate / max(self._rain_std * 3, 1.0))
                + 0.3 * current_soil
                + 0.3 * (rain_forecast / max(50.0, 1.0))
            )
            p_median = 1.0 / (1.0 + math.exp(-5.0 * (x - 0.5)))

            # Confidence interval widens with horizon
            spread = 0.05 + 0.03 * hours + 0.1 * (1.0 - min(len(rain_rate) / 18, 1.0))
            p_lower = max(0.0, p_median - spread)
            p_upper = min(1.0, p_median + spread)

            results[horizon] = (p_lower, p_median, p_upper, rain_forecast)

        return results

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Keras-like interface: predict P(flood) for batch of windows."""
        probs = []
        for i in range(X.shape[0]):
            horizons = self.predict_horizons(X[i])
            # Return 30-min forecast as default
            _, p_med, _, _ = horizons[Horizon.MIN_30]
            probs.append(p_med)
        return np.array(probs).reshape(-1, 1)


def build_forecast_lstm(
    seq_len: int = WINDOW_TIMESTEPS,
    n_features: int = N_FEATURES,
    n_horizons: int = 3,
    lstm_units: Optional[List[int]] = None,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
) -> Any:
    """
    Build a multi-output LSTM for simultaneous multi-horizon forecasting.

    Architecture:
    ─────────────
        Input(18, 8) — 3hr window × 8 features
        → LSTM(96, return_sequences=True)
        → Dropout(0.3)
        → LSTM(48)
        → Dropout(0.3)
        → Dense(32, relu)
        → 3 output heads:
            head_30min → Dense(3) → [p_lower, p_median, p_upper]
            head_1hr   → Dense(3) → [p_lower, p_median, p_upper]
            head_3hr   → Dense(3) → [p_lower, p_median, p_upper]

    Each head outputs 3 values for quantile regression:
        • τ=0.10 (lower bound)
        • τ=0.50 (median / point estimate)
        • τ=0.90 (upper bound)

    Loss: pinball (quantile) loss averaged across heads.

    Parameters
    ----------
    seq_len : int
        Input sequence length (timesteps).
    n_features : int
        Features per timestep.
    n_horizons : int
        Number of forecast horizons (default 3).
    lstm_units : list of int or None
        Units per LSTM layer. Default [96, 48].
    dropout : float
        Dropout rate.
    learning_rate : float
        Adam learning rate.

    Returns
    -------
    tf.keras.Model
        Multi-output model with 3 heads.
    """
    if lstm_units is None:
        lstm_units = [96, 48]

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        logger.warning("TensorFlow unavailable — using fallback forecaster")
        return None

    # Shared backbone
    inp = layers.Input(shape=(seq_len, n_features), name="input_window")

    x = layers.LSTM(
        lstm_units[0],
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="lstm_1",
    )(inp)
    x = layers.Dropout(dropout, name="drop_1")(x)

    x = layers.LSTM(
        lstm_units[1],
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="lstm_2",
    )(x)
    x = layers.Dropout(dropout, name="drop_2")(x)

    shared = layers.Dense(32, activation="relu", name="shared_dense")(x)

    # Per-horizon heads: each outputs 3 quantile predictions
    outputs = []
    for i, horizon in enumerate(ALL_HORIZONS):
        head = layers.Dense(16, activation="relu", name=f"head_{horizon.value}m_dense")(shared)
        head = layers.Dense(
            3, activation="sigmoid", name=f"head_{horizon.value}m_out"
        )(head)
        outputs.append(head)

    model = keras.Model(inputs=inp, outputs=outputs, name="flood_forecast_lstm")

    # Quantile loss (pinball loss)
    quantiles = [QUANTILE_LOWER, 0.5, QUANTILE_UPPER]

    def quantile_loss(y_true, y_pred):
        """Combined pinball loss for 3 quantiles."""
        total_loss = tf.constant(0.0)
        for j, tau in enumerate(quantiles):
            error = y_true - y_pred[:, j]
            total_loss += tf.reduce_mean(
                tf.maximum(tau * error, (tau - 1.0) * error)
            )
        return total_loss / len(quantiles)

    losses = {f"head_{h.value}m_out": quantile_loss for h in ALL_HORIZONS}

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
    )

    return model


# ===================================================================
#  5. TRAINING DATA GENERATION (SYNTHETIC)
# ===================================================================


def generate_forecast_training_data(
    n_samples: int = 3000,
    seq_len: int = WINDOW_TIMESTEPS,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[Horizon, np.ndarray]]:
    """
    Generate synthetic training data for the multi-horizon forecaster.

    Creates realistic rainfall sequences with embedded bursts, trends,
    and varying soil / atmospheric conditions.

    Returns
    -------
    X : np.ndarray, shape (n_samples, seq_len, N_FEATURES)
    y : dict mapping Horizon → np.ndarray of shape (n_samples,)
        Binary flood labels at each horizon.
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((n_samples, seq_len, N_FEATURES), dtype=np.float64)
    y = {h: np.zeros(n_samples) for h in ALL_HORIZONS}

    for i in range(n_samples):
        # Generate base rainfall sequence (exponential + bursts)
        base_rain = rng.exponential(2.0, seq_len).clip(0, 80)

        # 20% chance of an embedded burst
        if rng.rand() < 0.20:
            burst_start = rng.randint(seq_len // 3, 2 * seq_len // 3)
            burst_len = rng.randint(3, min(8, seq_len - burst_start))
            burst_intensity = rng.uniform(15, 60)
            base_rain[burst_start:burst_start + burst_len] += burst_intensity

        # Soil moisture: correlated with recent rain
        soil = 0.2 + 0.5 * (np.cumsum(base_rain) / (np.cumsum(base_rain).max() + 1))
        soil = soil.clip(0.1, 0.95) + rng.normal(0, 0.05, seq_len)
        soil = soil.clip(0.05, 0.99)

        # Other features
        temp = 28 + 5 * np.sin(np.arange(seq_len) * np.pi / 9) + rng.normal(0, 1, seq_len)
        humidity = 60 + 20 * soil + rng.normal(0, 3, seq_len)
        humidity = humidity.clip(30, 100)
        pressure = 1013 + np.cumsum(rng.normal(0, 0.2, seq_len))
        wind = rng.uniform(0, 15, seq_len)

        # Build feature matrix
        cumulative = np.cumsum(base_rain)
        rate = base_rain * (60.0 / OBS_INTERVAL_MIN)

        X[i, :, 0] = base_rain
        X[i, :, 1] = cumulative
        X[i, :, 2] = rate
        X[i, :, 3] = soil
        X[i, :, 4] = temp
        X[i, :, 5] = humidity
        X[i, :, 6] = pressure
        X[i, :, 7] = wind

        # Generate flood labels based on physics heuristic
        recent_rain = base_rain[-6:].sum()        # last hour rain
        total_rain = cumulative[-1]
        final_soil = soil[-1]
        max_rate = rate.max()

        for horizon in ALL_HORIZONS:
            # Longer horizon → harder to predict → use broader signal
            horizon_factor = horizon.value / 60.0
            flood_score = (
                0.3 * (recent_rain / 30.0)
                + 0.2 * (total_rain / 100.0)
                + 0.2 * final_soil
                + 0.2 * (max_rate / 60.0)
                + 0.1 * horizon_factor * (recent_rain / 20.0)
            )
            prob = 1.0 / (1.0 + math.exp(-8 * (flood_score - 0.5)))
            y[horizon][i] = 1.0 if rng.rand() < prob else 0.0

    return X, y


# ===================================================================
#  6. TRAINING
# ===================================================================


@dataclass
class ForecastTrainResult:
    """Training result for the forecast LSTM."""
    model: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    using_fallback: bool = False
    epochs_trained: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "using_fallback": self.using_fallback,
            "epochs_trained": self.epochs_trained,
            **{k: round(v, 4) for k, v in self.metrics.items()},
        }


def train_forecast_model(
    X: np.ndarray,
    y: Dict[Horizon, np.ndarray],
    epochs: int = 80,
    batch_size: int = 32,
    patience: int = 10,
    validation_split: float = 0.2,
) -> ForecastTrainResult:
    """
    Train the multi-horizon forecast LSTM.

    If TensorFlow is unavailable, trains the fallback forecaster instead.

    Parameters
    ----------
    X : shape (n_samples, seq_len, n_features)
    y : dict of Horizon → labels
    epochs : int
    batch_size : int
    patience : int
    validation_split : float

    Returns
    -------
    ForecastTrainResult
    """
    n = X.shape[0]
    split = int(n * (1 - validation_split))
    X_train, X_val = X[:split], X[split:]

    # Try Keras model first
    model = build_forecast_lstm(
        seq_len=X.shape[1],
        n_features=X.shape[2],
    )

    if model is None:
        # Fallback
        logger.info("Training fallback forecaster (no TensorFlow)")
        fallback = _FallbackForecaster()

        # For fallback, just use 30-min labels
        y_train = y[Horizon.MIN_30][:split]
        fallback.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score
        y_val_labels = y[Horizon.MIN_30][split:]
        preds = fallback.predict(X_val).flatten()
        y_pred = (preds >= 0.5).astype(int)
        acc = accuracy_score(y_val_labels, y_pred)

        return ForecastTrainResult(
            model=fallback,
            metrics={"accuracy_30min": float(acc)},
            using_fallback=True,
            epochs_trained=0,
        )

    # Keras training path
    import tensorflow as tf
    from tensorflow import keras

    y_train_dict = {
        f"head_{h.value}m_out": y[h][:split] for h in ALL_HORIZONS
    }
    y_val_dict = {
        f"head_{h.value}m_out": y[h][split:] for h in ALL_HORIZONS
    }

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    epochs_trained = len(history.history.get("loss", []))
    val_loss = min(history.history.get("val_loss", [0.0]))

    metrics = {
        "val_loss": float(val_loss),
        "epochs_trained": epochs_trained,
    }

    # Per-horizon accuracy
    predictions = model.predict(X_val, verbose=0)
    from sklearn.metrics import accuracy_score
    for idx, horizon in enumerate(ALL_HORIZONS):
        p_median = predictions[idx][:, 1]  # middle quantile
        y_pred = (p_median >= 0.5).astype(int)
        y_true = y[horizon][split:]
        acc = accuracy_score(y_true, y_pred)
        metrics[f"accuracy_{horizon.value}min"] = float(acc)
        logger.info("Horizon T+%dmin — accuracy=%.4f", horizon.value, acc)

    return ForecastTrainResult(
        model=model,
        metrics=metrics,
        using_fallback=False,
        epochs_trained=epochs_trained,
    )


# ===================================================================
#  7. ROLLING PREDICTION
# ===================================================================


def rolling_predict(
    model: Any,
    window: np.ndarray,
    is_fallback: bool = False,
) -> Dict[Horizon, Tuple[float, float, float, float]]:
    """
    Run the model on a single window and extract per-horizon predictions.

    Rolling prediction logic:
    ────────────────────────
    At each prediction cycle:
        1. The latest 18 observations form the input window.
        2. The window is normalised (z-score).
        3. The model produces 3 quantile estimates per horizon.
        4. The window is "rolled forward" by dropping the oldest
           observation and appending the newest one.

    This function handles a single cycle. The caller is responsible
    for advancing the window between cycles.

    Returns
    -------
    dict : Horizon → (p_lower, p_median, p_upper, rain_forecast_mm)
    """
    if is_fallback or isinstance(model, _FallbackForecaster):
        return model.predict_horizons(window)

    # Normalise
    norm_window, _, _ = normalise_window(window)

    # Add batch dimension: (1, seq_len, n_features)
    X = norm_window[np.newaxis, ...]

    # Predict: returns list of 3 arrays, each shape (1, 3)
    predictions = model.predict(X, verbose=0)

    results = {}
    rain_rate = float(window[-1, 2])  # current rate

    for idx, horizon in enumerate(ALL_HORIZONS):
        preds = predictions[idx][0]  # shape (3,)
        p_lower = float(np.clip(preds[0], 0, 1))
        p_median = float(np.clip(preds[1], 0, 1))
        p_upper = float(np.clip(preds[2], 0, 1))

        # Ensure ordering: lower ≤ median ≤ upper
        p_lower = min(p_lower, p_median)
        p_upper = max(p_upper, p_median)

        # Estimate rainfall accumulation to horizon
        hours = horizon.value / 60.0
        rain_forecast = rain_rate * hours

        results[horizon] = (p_lower, p_median, p_upper, rain_forecast)

    return results


# ===================================================================
#  8. CONFIDENCE INTERVAL ESTIMATION
# ===================================================================


def estimate_confidence(
    p_lower: float,
    p_median: float,
    p_upper: float,
    window_completeness: float = 1.0,
    burst_detected: bool = False,
) -> Dict[str, float]:
    """
    Refine confidence intervals with domain-specific adjustments.

    Confidence interval estimation strategy:
    ─────────────────────────────────────────
    1. **Quantile regression** (primary): The LSTM's 3-output heads directly
       estimate the 10th, 50th, and 90th percentiles of P(flood).

    2. **Window completeness penalty**: If fewer than 18 observations are
       available (cold start), confidence intervals are widened proportionally.

    3. **Burst amplification**: During detected rainfall bursts, the upper
       bound is pushed higher (flash floods have fat-tailed distributions).

    4. **Horizon scaling**: Longer horizons inherently have wider intervals
       (handled upstream by the model's learned uncertainty).

    Parameters
    ----------
    p_lower, p_median, p_upper : float
        Raw quantile predictions from the model.
    window_completeness : float
        Fraction of window that has real data (vs. zero-padding). ∈ [0, 1].
    burst_detected : bool
        Whether a rainfall burst was detected.

    Returns
    -------
    dict with adjusted p_lower, p_median, p_upper, confidence_score.
    """
    # 1. Width penalty for incomplete windows
    completeness_factor = max(window_completeness, 0.3)
    extra_spread = 0.08 * (1.0 - completeness_factor)
    adj_lower = max(0.0, p_lower - extra_spread)
    adj_upper = min(1.0, p_upper + extra_spread)

    # 2. Burst amplification: push upper bound up during bursts
    if burst_detected:
        burst_push = 0.10 * (1.0 - p_upper)  # proportional to headroom
        adj_upper = min(1.0, adj_upper + burst_push)

    # 3. Confidence score: narrow interval → high confidence
    width = adj_upper - adj_lower
    confidence_score = max(0.0, 1.0 - width)

    return {
        "p_lower": round(adj_lower, 4),
        "p_median": round(p_median, 4),
        "p_upper": round(adj_upper, 4),
        "interval_width": round(width, 4),
        "confidence_score": round(confidence_score, 4),
    }


# ===================================================================
#  9. ENSEMBLE COMBINATION WITH XGBOOST
# ===================================================================


def ensemble_with_xgboost(
    lstm_horizons: Dict[Horizon, Tuple[float, float, float, float]],
    xgb_probability: float,
    alpha: float = 0.55,
    xgb_confidence: float = 1.0,
    lstm_confidence: float = 1.0,
) -> Dict[Horizon, Dict[str, float]]:
    """
    Combine LSTM multi-horizon forecasts with XGBoost snapshot prediction.

    Ensemble combination strategy:
    ──────────────────────────────
    XGBoost operates on a **static feature vector** (current snapshot),
    providing a "what's the flood risk RIGHT NOW?" estimate. The LSTM
    operates on a **temporal window**, providing "what's the flood risk
    IN THE FUTURE?" estimates.

    Combination:
        P_final(t+h) = α_eff · P_xgb + (1 - α_eff) · P_lstm(t+h)

    Where α_eff decreases with horizon:
        • T+30min: α_eff = α × 1.0  (XGBoost highly relevant)
        • T+1hr:   α_eff = α × 0.7  (XGBoost partially relevant)
        • T+3hr:   α_eff = α × 0.3  (LSTM dominates for far future)

    Rationale: XGBoost's tabular snapshot is most informative for
    near-term predictions (current soil moisture, pressure, etc.
    directly affect 30-min outlook). For 3-hour forecasts, the
    temporal trajectory matters more than the current snapshot.

    Parameters
    ----------
    lstm_horizons : dict
        Horizon → (p_lower, p_median, p_upper, rain_mm).
    xgb_probability : float
        XGBoost's current flood probability.
    alpha : float
        Base blending weight for XGBoost.
    xgb_confidence, lstm_confidence : float
        Confidence scores (0–1) for dynamic weighting.

    Returns
    -------
    dict : Horizon → {ensemble_prob, xgb_contrib, lstm_contrib, ...}
    """
    # Horizon-dependent alpha decay
    ALPHA_DECAY: Dict[Horizon, float] = {
        Horizon.MIN_30: 1.0,
        Horizon.HOUR_1: 0.7,
        Horizon.HOUR_3: 0.3,
    }

    # Dynamic weighting by confidence
    total_conf = xgb_confidence + lstm_confidence + 1e-8
    conf_w_xgb = xgb_confidence / total_conf
    conf_w_lstm = lstm_confidence / total_conf

    results = {}
    for horizon in ALL_HORIZONS:
        p_lower, p_median, p_upper, rain_mm = lstm_horizons.get(
            horizon, (0.0, 0.0, 0.0, 0.0)
        )

        # Effective alpha for this horizon
        alpha_eff = alpha * ALPHA_DECAY[horizon]

        # Apply confidence weighting on top
        alpha_final = alpha_eff * conf_w_xgb / (
            alpha_eff * conf_w_xgb + (1 - alpha_eff) * conf_w_lstm + 1e-8
        )

        # Blend
        p_ensemble = alpha_final * xgb_probability + (1 - alpha_final) * p_median
        p_ensemble = float(np.clip(p_ensemble, 0.0, 1.0))

        # Adjust confidence interval to account for ensemble
        ens_lower = alpha_final * xgb_probability + (1 - alpha_final) * p_lower
        ens_upper = alpha_final * xgb_probability + (1 - alpha_final) * p_upper
        ens_lower = float(np.clip(ens_lower, 0.0, 1.0))
        ens_upper = float(np.clip(ens_upper, 0.0, 1.0))

        # Risk classification
        risk = _classify_risk(p_ensemble)

        results[horizon] = {
            "ensemble_probability": round(p_ensemble, 4),
            "confidence_lower": round(ens_lower, 4),
            "confidence_upper": round(ens_upper, 4),
            "xgb_probability": round(xgb_probability, 4),
            "lstm_probability": round(p_median, 4),
            "alpha_effective": round(alpha_final, 4),
            "rainfall_forecast_mm": round(rain_mm, 2),
            "risk_level": risk,
        }

    return results


def _classify_risk(prob: float) -> str:
    """Map probability to risk level string."""
    if prob < 0.15:
        return "minimal"
    elif prob < 0.35:
        return "low"
    elif prob < 0.60:
        return "moderate"
    elif prob < 0.80:
        return "high"
    else:
        return "critical"


# ===================================================================
#  10. COMPLETE FORECAST PIPELINE
# ===================================================================


def run_forecast(
    observations: List[Observation],
    model: Any = None,
    xgb_probability: float = 0.0,
    alpha: float = 0.55,
    is_fallback: bool = True,
) -> ForecastResult:
    """
    End-to-end sub-hour flood forecast.

    Full pipeline:
        1. Build input window from observations.
        2. Detect rainfall bursts.
        3. Estimate rainfall trend.
        4. Run LSTM model for multi-horizon predictions.
        5. Apply confidence interval adjustments.
        6. Combine with XGBoost ensemble.
        7. Package into ForecastResult.

    Parameters
    ----------
    observations : list of Observation
        Raw 10-minute readings (oldest first).
    model : Any
        Trained LSTM model or fallback predictor.
    xgb_probability : float
        XGBoost's current snapshot flood probability.
    alpha : float
        Base XGBoost blending weight.
    is_fallback : bool
        Whether the model is the fallback predictor.

    Returns
    -------
    ForecastResult
    """
    # 1. Build window
    window = build_input_window(observations)
    window_completeness = min(len(observations) / WINDOW_TIMESTEPS, 1.0)

    # 2. Burst detection
    burst = detect_burst(window)

    # 3. Trend estimation
    trend = estimate_trend(window)

    # 4. Model prediction
    if model is None:
        model = _FallbackForecaster()
        model.fit(window[np.newaxis, ...], np.zeros(1))
        is_fallback = True

    lstm_horizons = rolling_predict(model, window, is_fallback=is_fallback)

    # 5. Confidence adjustment
    adjusted_horizons = {}
    for horizon, (p_lo, p_med, p_hi, rain_mm) in lstm_horizons.items():
        conf = estimate_confidence(
            p_lo, p_med, p_hi,
            window_completeness=window_completeness,
            burst_detected=burst.is_burst,
        )
        adjusted_horizons[horizon] = (
            conf["p_lower"],
            conf["p_median"],
            conf["p_upper"],
            rain_mm,
        )

    # 6. Ensemble with XGBoost
    ensemble_results = ensemble_with_xgboost(
        lstm_horizons=adjusted_horizons,
        xgb_probability=xgb_probability,
        alpha=alpha,
    )

    # 7. Package
    forecasts = []
    for horizon in ALL_HORIZONS:
        ens = ensemble_results[horizon]
        forecasts.append(HorizonForecast(
            horizon=horizon,
            horizon_label=HORIZON_LABELS[horizon],
            flood_probability=ens["ensemble_probability"],
            confidence_lower=ens["confidence_lower"],
            confidence_upper=ens["confidence_upper"],
            confidence_width=round(ens["confidence_upper"] - ens["confidence_lower"], 4),
            rainfall_forecast_mm=ens["rainfall_forecast_mm"],
            risk_level=ens["risk_level"],
        ))

    # Window statistics
    rain_total = float(window[:, 1].max())  # cumulative rainfall
    rain_max_rate = float(window[:, 2].max())
    soil_current = float(window[-1, 3])

    window_stats = {
        "observations_count": len(observations),
        "window_completeness": round(window_completeness, 2),
        "total_rainfall_mm": round(rain_total, 2),
        "max_rainfall_rate_mm_hr": round(rain_max_rate, 2),
        "current_soil_moisture": round(soil_current, 3),
    }

    return ForecastResult(
        forecasts=forecasts,
        burst_detection=burst,
        trend_estimate=trend,
        ensemble_details={
            "xgb_probability": round(xgb_probability, 4),
            "base_alpha": round(alpha, 3),
        },
        window_stats=window_stats,
        model_type="fallback" if is_fallback else "lstm_multi_horizon",
    )
