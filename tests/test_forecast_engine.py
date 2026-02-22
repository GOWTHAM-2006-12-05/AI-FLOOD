"""End-to-end tests for the sub-hour flood forecasting engine."""

from __future__ import annotations

import sys, os, math, time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backend.app.ml.forecast_engine import (
    ALL_HORIZONS,
    WINDOW_TIMESTEPS,
    N_FEATURES,
    Horizon,
    Observation,
    BurstDetection,
    TrendEstimate,
    HorizonForecast,
    ForecastResult,
    _FallbackForecaster,
    build_input_window,
    normalise_window,
    detect_burst,
    estimate_trend,
    rolling_predict,
    estimate_confidence,
    ensemble_with_xgboost,
    run_forecast,
    generate_forecast_training_data,
    train_forecast_model,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ── Helper: generate test observations ────────────────────────────

def make_observations(
    n: int = 18,
    base_rain: float = 2.0,
    burst_at: int = -1,
    burst_intensity: float = 30.0,
) -> list:
    """Generate n synthetic observations at 10-min intervals."""
    obs = []
    t0 = 1740200000.0
    for i in range(n):
        rain = base_rain + (burst_intensity if i == burst_at else 0.0)
        obs.append(Observation(
            timestamp_unix=t0 + i * 600,
            rainfall_mm=rain,
            soil_moisture=0.3 + 0.02 * i,
            temperature_c=28.0 + 0.1 * i,
            relative_humidity=70.0 + 0.5 * i,
            surface_pressure_hpa=1013.0 - 0.1 * i,
            wind_speed_ms=3.0 + 0.2 * i,
        ))
    return obs


# ── 1. Input window construction ─────────────────────────────────

print("\n=== 1. INPUT WINDOW CONSTRUCTION ===\n")

obs_full = make_observations(18)
w = build_input_window(obs_full)
check("full window shape", w.shape == (18, N_FEATURES), f"got {w.shape}")
check("rainfall_mm > 0", w[-1, 0] > 0, f"got {w[-1, 0]}")
check("cumulative monotonic", all(w[i, 1] <= w[i+1, 1] for i in range(17)))
check("rate = rainfall × 6", abs(w[0, 2] - w[0, 0] * 6) < 0.01)

# Cold start: fewer than 18 obs → left-padded
obs_short = make_observations(5)
w_short = build_input_window(obs_short)
check("short window shape", w_short.shape == (18, N_FEATURES))
check("short window left-padded", w_short[0, 0] == 0.0 and w_short[13, 0] > 0)

# Normalisation
w_norm, means, stds = normalise_window(w)
check("normalised shape", w_norm.shape == w.shape)
check("normalised mean ≈ 0", abs(w_norm.mean()) < 0.5)


# ── 2. Burst detection ───────────────────────────────────────────

print("\n=== 2. BURST DETECTION ===\n")

# No burst in calm rain
obs_calm = make_observations(18, base_rain=1.0)
w_calm = build_input_window(obs_calm)
burst_calm = detect_burst(w_calm)
check("calm rain: no burst", not burst_calm.is_burst)

# Burst in heavy rain
obs_burst = make_observations(18, base_rain=1.0, burst_at=10, burst_intensity=40.0)
w_burst = build_input_window(obs_burst)
burst_heavy = detect_burst(w_burst)
check("heavy burst detected", burst_heavy.is_burst)
check("burst intensity > 0", burst_heavy.burst_intensity_mm_hr > 0)
check("max rate > threshold", burst_heavy.max_rate_mm_hr > 20)

# Sustained heavy rain
obs_sustained = make_observations(18, base_rain=5.0)  # 5mm → 30mm/hr
w_sustained = build_input_window(obs_sustained)
burst_sustained = detect_burst(w_sustained)
check("sustained heavy: burst", burst_sustained.is_burst)

# to_dict works
d = burst_heavy.to_dict()
check("burst dict has keys", "is_burst" in d and "burst_duration_minutes" in d)


# ── 3. Trend estimation ─────────────────────────────────────────

print("\n=== 3. TREND ESTIMATION ===\n")

# Increasing rain → intensifying
obs_increasing = []
t0 = 1740200000.0
for i in range(18):
    obs_increasing.append(Observation(
        timestamp_unix=t0 + i * 600,
        rainfall_mm=1.0 + i * 2.0,  # increasing
    ))
w_inc = build_input_window(obs_increasing)
trend_inc = estimate_trend(w_inc)
check("increasing trend slope > 0", trend_inc.slope > 0, f"got {trend_inc.slope}")
check("projected rate > 0", trend_inc.projected_rate_mm_hr > 0)

# Decreasing rain rate → negative slope
# Ensure the last 6 points show a clear downward trend (no bottoming-out)
w_dec = np.zeros((18, N_FEATURES))
for i in range(18):
    rate = 80.0 - i * 4.0  # 80 → 12 mm/hr — never bottoms out
    w_dec[i, 0] = rate / 6.0
    w_dec[i, 1] = sum((80.0 - j * 4.0) / 6.0 for j in range(i + 1))
    w_dec[i, 2] = rate
    w_dec[i, 3] = 0.4
    w_dec[i, 4] = 28.0
    w_dec[i, 5] = 70.0
    w_dec[i, 6] = 1013.0
    w_dec[i, 7] = 3.0
trend_dec = estimate_trend(w_dec)
check("decreasing trend slope < 0", trend_dec.slope < 0,
      f"got slope={trend_dec.slope}")

# Stable rain
obs_stable = make_observations(18, base_rain=3.0)
w_stable = build_input_window(obs_stable)
trend_stable = estimate_trend(w_stable)
check("stable trend direction", trend_stable.trend_direction == "stable",
      f"got {trend_stable.trend_direction}")

# R² reasonably high for clean trends
check("R² > 0.5 for increasing", trend_inc.r_squared > 0.5, f"got {trend_inc.r_squared}")


# ── 4. Fallback forecaster ──────────────────────────────────────

print("\n=== 4. FALLBACK FORECASTER ===\n")

fallback = _FallbackForecaster(seed=42)
X_dummy = build_input_window(make_observations(18, base_rain=5.0))
fallback.fit(X_dummy[None, ...], None)  # fit with dummy data

horizons = fallback.predict_horizons(X_dummy)
check("3 horizons returned", len(horizons) == 3)

for h in ALL_HORIZONS:
    p_lo, p_med, p_hi, rain_mm = horizons[h]
    check(f"H={h.value}min: lo≤med≤hi",
          p_lo <= p_med <= p_hi,
          f"lo={p_lo:.3f} med={p_med:.3f} hi={p_hi:.3f}")
    check(f"H={h.value}min: probs in [0,1]",
          0 <= p_lo and p_hi <= 1)
    check(f"H={h.value}min: rain forecast ≥ 0", rain_mm >= 0)

# Keras-like predict interface
preds = fallback.predict(X_dummy[None, ...])
check("predict shape (1,1)", preds.shape == (1, 1))


# ── 5. Rolling prediction ───────────────────────────────────────

print("\n=== 5. ROLLING PREDICTION ===\n")

obs_for_rolling = make_observations(18, base_rain=4.0)
w_roll = build_input_window(obs_for_rolling)

results = rolling_predict(fallback, w_roll, is_fallback=True)
check("rolling returns 3 horizons", len(results) == 3)

for h in ALL_HORIZONS:
    p_lo, p_med, p_hi, rain_mm = results[h]
    check(f"rolling H={h.value}min valid",
          0 <= p_lo <= p_med <= p_hi <= 1,
          f"lo={p_lo:.3f} med={p_med:.3f} hi={p_hi:.3f}")


# ── 6. Confidence interval estimation ────────────────────────────

print("\n=== 6. CONFIDENCE INTERVALS ===\n")

# Full window
conf = estimate_confidence(0.2, 0.4, 0.6, window_completeness=1.0, burst_detected=False)
check("conf keys present", all(k in conf for k in ["p_lower", "p_median", "p_upper", "confidence_score"]))
check("conf lower ≤ upper", conf["p_lower"] <= conf["p_upper"])

# Incomplete window → wider interval
conf_short = estimate_confidence(0.2, 0.4, 0.6, window_completeness=0.3, burst_detected=False)
check("incomplete → wider", conf_short["interval_width"] > conf["interval_width"],
      f"short={conf_short['interval_width']:.3f} full={conf['interval_width']:.3f}")

# Burst → upper pushed higher
conf_burst = estimate_confidence(0.2, 0.4, 0.6, window_completeness=1.0, burst_detected=True)
check("burst → higher upper", conf_burst["p_upper"] >= conf["p_upper"],
      f"burst={conf_burst['p_upper']:.3f} normal={conf['p_upper']:.3f}")


# ── 7. XGBoost ensemble ─────────────────────────────────────────

print("\n=== 7. ENSEMBLE WITH XGBOOST ===\n")

lstm_horizons = {
    Horizon.MIN_30: (0.2, 0.35, 0.5, 5.0),
    Horizon.HOUR_1: (0.25, 0.45, 0.65, 12.0),
    Horizon.HOUR_3: (0.35, 0.60, 0.80, 30.0),
}

ens = ensemble_with_xgboost(
    lstm_horizons=lstm_horizons,
    xgb_probability=0.40,
    alpha=0.55,
)

check("ensemble returns 3 horizons", len(ens) == 3)

for h in ALL_HORIZONS:
    r = ens[h]
    check(f"ens H={h.value}min prob in [0,1]",
          0 <= r["ensemble_probability"] <= 1)
    check(f"ens H={h.value}min has risk_level",
          r["risk_level"] in ["minimal", "low", "moderate", "high", "critical"])

# XGBoost weight should decrease with horizon
a30 = ens[Horizon.MIN_30]["alpha_effective"]
a60 = ens[Horizon.HOUR_1]["alpha_effective"]
a180 = ens[Horizon.HOUR_3]["alpha_effective"]
check("alpha decay: 30min > 1hr > 3hr",
      a30 > a60 > a180,
      f"30min={a30:.3f} 1hr={a60:.3f} 3hr={a180:.3f}")


# ── 8. Full pipeline ────────────────────────────────────────────

print("\n=== 8. FULL FORECAST PIPELINE ===\n")

obs_full_test = make_observations(18, base_rain=3.0)
result = run_forecast(
    observations=obs_full_test,
    model=None,  # will use fallback
    xgb_probability=0.30,
    alpha=0.50,
)

check("result has 3 forecasts", len(result.forecasts) == 3)
check("burst detection present", result.burst_detection is not None)
check("trend estimate present", result.trend_estimate is not None)
check("window stats present", "observations_count" in result.window_stats)
check("model type is fallback", result.model_type == "fallback")

# Verify to_dict serialisation
d = result.to_dict()
check("to_dict has forecasts", "forecasts" in d and len(d["forecasts"]) == 3)
check("to_dict forecasts have CI", "confidence_interval" in d["forecasts"][0])

# Higher rain → higher probability
obs_heavy = make_observations(18, base_rain=10.0)
result_heavy = run_forecast(obs_heavy, xgb_probability=0.6)
result_light = run_forecast(make_observations(18, base_rain=0.5), xgb_probability=0.1)

p_heavy_30 = result_heavy.forecasts[0].flood_probability
p_light_30 = result_light.forecasts[0].flood_probability
check("heavy rain > light rain P(flood)",
      p_heavy_30 > p_light_30,
      f"heavy={p_heavy_30:.3f} light={p_light_30:.3f}")


# ── 9. Training (synthetic data) ────────────────────────────────

print("\n=== 9. TRAINING DATA GENERATION ===\n")

X, y = generate_forecast_training_data(n_samples=500, seed=99)
check("X shape", X.shape == (500, 18, N_FEATURES), f"got {X.shape}")
check("y has 3 horizons", len(y) == 3)
for h in ALL_HORIZONS:
    check(f"y[{h.value}min] shape", y[h].shape == (500,), f"got {y[h].shape}")
    check(f"y[{h.value}min] binary", set(np.unique(y[h])).issubset({0.0, 1.0}))

# Verify some positive labels exist
for h in ALL_HORIZONS:
    pos_rate = y[h].mean()
    check(f"y[{h.value}min] pos rate {pos_rate:.2f}", 0.05 < pos_rate < 0.95)


print("\n=== 10. TRAIN FALLBACK MODEL ===\n")

result = train_forecast_model(X, y, epochs=10, batch_size=64)
check("training returned result", result.model is not None)
check("trained model is fallback", result.using_fallback)  # TF likely not installed
check("metrics populated", "accuracy_30min" in result.metrics)


# ── Summary ───────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"RESULTS:  {passed} passed,  {failed} failed,  {passed+failed} total")
print(f"{'='*50}\n")

sys.exit(0 if failed == 0 else 1)
