# Sub-Hour Flood Forecasting Engine

## Overview

The forecasting engine produces multi-horizon flood probability predictions at
**T+30 min**, **T+1 hr**, and **T+3 hr**, using a combination of LSTM temporal
modelling and XGBoost snapshot analysis. Every prediction includes confidence
intervals estimated via quantile regression.

```
┌───────────────────────────────────────────────────────────────────┐
│                  SUB-HOUR FORECAST PIPELINE                      │
│                                                                   │
│  10-min sensor readings (last 3 hours)                           │
│     ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬···┬───┐      │
│     │ t₁│ t₂│ t₃│ t₄│ t₅│ t₆│ t₇│ t₈│ t₉│t₁₀│t₁₁│   │t₁₈│     │
│     └─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴──┬─┴──┬─···┴──┬┘     │
│       └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴──···┘       │
│                         │                                    │
│          ┌──────────────┼──────────────┐                     │
│          ▼              ▼              ▼                     │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│   │   BURST    │ │   TREND    │ │   LSTM     │              │
│   │  DETECTOR  │ │ ESTIMATOR  │ │  FORECAST  │              │
│   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘              │
│         │              │              │                      │
│         └──────────────┼──────────────┘                      │
│                        ▼                                     │
│              ┌──────────────────┐     ┌──────────────┐       │
│              │   CONFIDENCE     │◄────│   XGBoost    │       │
│              │   ADJUSTMENT     │     │  (snapshot)   │       │
│              └────────┬─────────┘     └──────────────┘       │
│                       ▼                                      │
│              ┌──────────────────┐                            │
│              │    ENSEMBLE      │                            │
│              │    COMBINER      │                            │
│              └────────┬─────────┘                            │
│                       ▼                                      │
│    ┌──────────────────────────────────────────┐              │
│    │  T+30min      T+1hr        T+3hr        │              │
│    │  P=0.35       P=0.48       P=0.62       │              │
│    │  [0.22,0.47]  [0.31,0.64]  [0.40,0.81]  │              │
│    │  LOW          MODERATE     HIGH          │              │
│    └──────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────────────┘
```

---

## 1. Time-Series Input Window Design

### Why 10-Minute Intervals?

Sub-hour forecasting requires finer temporal resolution than the hourly data
used by traditional models. Flash floods can develop within 15–30 minutes;
hourly data misses the critical onset entirely.

```
Hourly data:        ─────┬─────────────┬─────────────┬──────
                         10mm          25mm          5mm

10-min data:        ──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
                      1  1  2  3  3  2  5  8 10  5  2  1
                                        ▲
                                    BURST DETECTED
                                   (not visible at
                                    hourly scale)
```

### Window Structure

| Parameter | Value | Rationale |
|---|---|---|
| **Interval** | 10 minutes | Captures rapid rainfall changes |
| **Window length** | 18 timesteps | 3 hours — matches longest horizon |
| **Total duration** | 180 minutes | Enough context for T+3hr forecast |

### Feature Vector (8 features per timestep)

| # | Feature | Unit | Why |
|---|---|---|---|
| 0 | `rainfall_mm` | mm | Raw rainfall in this interval |
| 1 | `rainfall_cumulative` | mm | Running total since window start — monotonic trend signal |
| 2 | `rainfall_rate_mm_hr` | mm/hr | Normalised intensity — comparable across intervals |
| 3 | `soil_moisture` | 0–1 | Absorption capacity (saturated soil = more runoff) |
| 4 | `temperature_c` | °C | Convective potential indicator |
| 5 | `relative_humidity` | % | Atmospheric moisture loading |
| 6 | `surface_pressure_hpa` | hPa | Pressure tendency signals approaching storms |
| 7 | `wind_speed_ms` | m/s | Storm intensity proxy |

### Why These Specific Features?

**Rainfall triplet** (raw + cumulative + rate): Instead of just one rainfall
field, three representations let the LSTM learn different temporal patterns:
- Raw captures burst peaks.
- Cumulative captures total loading.
- Rate normalises for comparison with thresholds.

**Soil moisture**: The same rainfall on dry soil (S=0.2) vs. saturated soil
(S=0.9) produces vastly different flood risks. Without this, the model can
only reason about rainfall volume.

**Pressure & wind**: Precursors — dropping pressure and rising wind often
*precede* rainfall intensification by 30–60 minutes, giving the model
predictive lead time beyond simple rainfall extrapolation.

### Input Tensor Shape

```
X.shape = (batch_size, 18, 8)
           │          │   └── 8 features per timestep
           │          └────── 18 timesteps (3 hours)
           └───────────────── batch of prediction requests
```

### Cold Start Handling

When fewer than 18 observations are available (system just started, sensor
outage), the window is **left-padded with zeros**:

```
Available:  [obs₁, obs₂, obs₃, obs₄, obs₅]     (5 observations)
Window:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, obs₁, ..., obs₅]
                                                     │← padded ─→│
Completeness: 5/18 = 0.28 → confidence intervals widened
```

---

## 2. Rolling Prediction Logic

### How It Works

Each prediction cycle:

```
Time ─────────────────────────────────────────────────────►

     ┌── Window at t=0 ──────────────────────────┐
     │  [obs₁  obs₂  obs₃  ...  obs₁₆  obs₁₇  obs₁₈]  │
     └────────────────────────────────────────────┘
          │                                        │
          ▼                                        ▼
     Predict T+30/T+60/T+180 from this window

          ···10 min later···

              ┌── Window at t=10min ────────────────────┐
              │  [obs₂  obs₃  obs₄  ...  obs₁₇  obs₁₈  obs₁₉]  │
              └──────────────────────────────────────────┘
              │  DROP oldest              APPEND newest  │
              ▼                                          ▼
         Predict T+30/T+60/T+180 from updated window
```

### Step-by-Step Process

1. **Ingest** the latest 10-minute observation (append to buffer).
2. **Slide** the window: take the most recent 18 observations.
3. **Build** the feature matrix: compute cumulative rainfall, rate.
4. **Normalise** with z-score (per-feature mean/std from window).
5. **Predict** all three horizons simultaneously.
6. **Adjust** confidence intervals based on data quality.
7. **Blend** with XGBoost's current prediction.
8. **Emit** the full forecast result.

### Prediction Frequency

| Scenario | Cycle Frequency |
|---|---|
| Normal monitoring | Every 10 minutes |
| Burst detected | Every 5 minutes (interpolate) |
| Critical risk | Continuous (every new reading) |

### Statefulness

The engine is **stateless by design**. All temporal context is encoded in the
18-step input window. This means:
- No hidden state to corrupt or drift.
- Any server can handle any prediction request.
- Easy horizontal scaling — no sticky sessions.

---

## 3. Confidence Interval Estimation

### Three-Layer Strategy

Confidence intervals come from three independent sources that are combined:

```
Layer 1: QUANTILE REGRESSION (primary)
──────────────────────────────────────
The LSTM has 3 output heads per horizon, each trained with
pinball (quantile) loss at τ = 0.10, 0.50, 0.90.

    Head output: [P_10%, P_50%, P_90%]

This directly learns the conditional distribution shape
from data, capturing heteroscedastic uncertainty (intervals
widen when the model is uncertain).


Layer 2: WINDOW COMPLETENESS PENALTY
──────────────────────────────────────
If fewer than 18 observations are available:

    extra_spread = 0.08 × (1 - completeness)
    P_lower -= extra_spread
    P_upper += extra_spread

5/18 observations → +6.4% interval widening on each side.


Layer 3: BURST AMPLIFICATION
──────────────────────────────────────
During detected bursts, the upper bound is pushed higher:

    burst_push = 0.10 × (1 - P_upper)

This reflects the fat-tailed distribution of flash flood
events: when a burst is occurring, extreme outcomes become
disproportionately more likely.
```

### Why Quantile Regression?

Alternative approaches and why quantile regression is preferred:

| Approach | Pros | Cons | Used? |
|---|---|---|---|
| **Quantile regression** | Learns data-driven intervals, captures asymmetric uncertainty | Requires 3× output neurons | **Yes** (primary) |
| MC Dropout | Uses existing model, no extra training | Slow (N forward passes), often overconfident | Future enhancement |
| Bootstrap ensemble | Well-calibrated intervals | Requires training N models | Too expensive |
| Gaussian assumption | Simple formula | Wrong assumption (flood risk is non-Gaussian) | No |

### Pinball Loss Function

For quantile $\tau$:

$$L_\tau(y, \hat{y}) = \begin{cases}
\tau \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\
(1 - \tau) \cdot (\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}$$

At $\tau = 0.90$, under-predictions are penalised 9× more than
over-predictions, pushing the model to output a value that 90% of
true observations fall below.

### Confidence Score

A scalar summary of interval tightness:

$$\text{confidence} = 1 - (P_{upper} - P_{lower})$$

| Confidence | Interval Width | Interpretation |
|---|---|---|
| > 0.85 | < 0.15 | High confidence |
| 0.70–0.85 | 0.15–0.30 | Moderate confidence |
| 0.50–0.70 | 0.30–0.50 | Low confidence |
| < 0.50 | > 0.50 | Very uncertain |

---

## 4. Ensemble Combination with XGBoost

### Why Ensemble?

XGBoost and LSTM capture complementary information:

```
┌─────────────────────────────────┬──────────────────────────────────┐
│         XGBoost                 │           LSTM                   │
│                                 │                                  │
│  ✓ Static snapshot              │  ✓ Temporal sequence             │
│  ✓ Feature interactions         │  ✓ Trend detection               │
│  ✓ "What is the risk NOW?"     │  ✓ "What WILL the risk be?"     │
│  ✓ Fast inference               │  ✓ Captures momentum             │
│  ✓ Interpretable (SHAP)        │  ✓ Non-linear time dynamics      │
│                                 │                                  │
│  ✗ Cannot see temporal trends  │  ✗ Slower inference              │
│  ✗ Blind to momentum           │  ✗ Less interpretable            │
│  ✗ No forecast horizon         │  ✗ Needs sequential data         │
└─────────────────────────────────┴──────────────────────────────────┘
```

### Horizon-Dependent Alpha Decay

The key insight: XGBoost's snapshot becomes less relevant for longer horizons.

$$\alpha_{eff}(h) = \alpha_{base} \times \delta(h)$$

| Horizon | Decay $\delta$ | Effective $\alpha$ (base=0.55) | Reasoning |
|---|---|---|---|
| T+30 min | 1.0 | 0.55 | Current conditions directly predict 30-min |
| T+1 hr | 0.7 | 0.385 | Conditions shifting, LSTM more valuable |
| T+3 hr | 0.3 | 0.165 | 3 hours out, temporal dynamics dominate |

### Dynamic Confidence Weighting

When both models provide confidence scores:

$$w_{xgb} = \frac{c_{xgb}}{c_{xgb} + c_{lstm} + \epsilon}$$

$$\alpha_{final} = \frac{\alpha_{eff} \cdot w_{xgb}}{\alpha_{eff} \cdot w_{xgb} + (1 - \alpha_{eff}) \cdot w_{lstm} + \epsilon}$$

This means:
- If XGBoost is confident and LSTM is not → XGBoost gets more weight.
- If LSTM is confident and XGBoost is not → LSTM gets more weight.
- If both are equally confident → reverts to default α.

### Final Ensemble Formula

$$P_{flood}(t+h) = \alpha_{final}(h) \cdot P_{xgb} + (1 - \alpha_{final}(h)) \cdot P_{lstm}(t+h)$$

Confidence intervals are also blended:

$$CI_{lower}(h) = \alpha_{final} \cdot P_{xgb} + (1 - \alpha_{final}) \cdot P_{lstm,10\%}(h)$$
$$CI_{upper}(h) = \alpha_{final} \cdot P_{xgb} + (1 - \alpha_{final}) \cdot P_{lstm,90\%}(h)$$

---

## 5. Rainfall Burst Detection

### Algorithm

```
Step 1: Extract rainfall rate column from window
Step 2: Compute step-over-step gradients: Δr(t) = r(t) - r(t-1)
Step 3: Find consecutive runs where rate > threshold (20 mm/hr)
Step 4: Check for gradient spikes > 5 mm/hr per step

BURST = (consecutive steps ≥ 2) OR (gradient spike detected)
```

### Why It Matters

Bursts trigger:
1. **Confidence interval widening** (upper bound pushed higher).
2. **Shortened prediction cycles** (5-min instead of 10-min).
3. **Alert escalation** in the downstream notification system.

### Key Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `rate_threshold` | 20 mm/hr | Rainfall rate to flag as intense |
| `gradient_threshold` | 5 mm/hr/step | Rate increase that signals acceleration |
| `sustained_steps` | 2 | Min consecutive steps above threshold |

---

## 6. Rainfall Trend Estimation

### Quadratic Regression

Fits a polynomial to the last 6 rate observations:

$$rate(t) = a \cdot t^2 + b \cdot t + c$$

From this:
- **Slope** (1st derivative at last point) = $2at_{last} + b$ → mm/hr per step.
- **Acceleration** (2nd derivative) = $2a$ → mm/hr² per step.

### Trend Classification

| Acceleration | Direction | Meaning |
|---|---|---|
| > +1.0 | Intensifying | Rain getting heavier faster — flash flood risk |
| -1.0 to +1.0 | Stable | Steady rainfall pattern |
| < -1.0 | Weakening | Rain easing — flood risk likely decreasing |

### Extrapolation

The fitted polynomial projects the rate one step (10 minutes) ahead.
This projected rate feeds into the LSTM as an auxiliary signal and
informs the confidence interval for the T+30min horizon.

---

## 7. API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/forecast/predict` | Full multi-horizon forecast |
| `POST` | `/api/v1/forecast/burst` | Burst detection only |
| `POST` | `/api/v1/forecast/trend` | Trend analysis only |
| `POST` | `/api/v1/forecast/train` | Train the forecast model |
| `GET` | `/api/v1/forecast/horizons` | List horizons & metadata |

### Example: Full Forecast

```json
POST /api/v1/forecast/predict
{
    "observations": [
        {"timestamp_unix": 1740200000, "rainfall_mm": 1.2, "soil_moisture": 0.4},
        {"timestamp_unix": 1740200600, "rainfall_mm": 2.5, "soil_moisture": 0.42},
        {"timestamp_unix": 1740201200, "rainfall_mm": 4.8, "soil_moisture": 0.45},
        {"timestamp_unix": 1740201800, "rainfall_mm": 8.1, "soil_moisture": 0.50},
        {"timestamp_unix": 1740202400, "rainfall_mm": 6.2, "soil_moisture": 0.55}
    ],
    "xgb_probability": 0.35,
    "alpha": 0.55
}
```

---

## 8. Module File Map

| File | Role |
|---|---|
| `backend/app/ml/forecast_engine.py` | Core engine: window, burst, trend, LSTM, ensemble, pipeline |
| `backend/app/api/v1/forecast.py` | FastAPI REST endpoints |
| `docs/SUB_HOUR_FORECASTING.md` | This document |
