# Unified Disaster Risk Aggregation Engine

## Overview

The Risk Aggregation Engine combines three independent hazard assessments into a **single composite risk score** (0–100%) and **risk level** (Safe / Watch / Warning / Severe). It is the central decision layer that drives alert routing, UI colour coding, and evacuation recommendations.

| Hazard       | Source Service           | Raw Output              | Normalised To |
|------------- |--------------------------|-------------------------|---------------|
| **Flood**    | `flood_service.py`       | Probability ∈ [0, 1]    | [0, 1]        |
| **Earthquake** | `earthquake_service.py` | Magnitude × depth factor | [0, 1]       |
| **Cyclone**  | `cyclone_service.py`     | Composite score ∈ [0, 1] | [0, 1]       |

---

## Mathematical Weighting Formula

### Step 1 — Normalisation

Each hazard is mapped to a comparable **[0, 1]** scale:

$$S_{flood} = \text{clamp}(P_{flood},\; 0,\; 1)$$

$$S_{earthquake} = \text{clamp}\!\left(\frac{M \times \text{depth\_severity\_factor}(d)}{10},\; 0,\; 1\right)$$

$$S_{cyclone} = \text{clamp}(R_{cyclone},\; 0,\; 1)$$

Where the **depth severity factor** is:
- Very shallow (< 10 km): **×1.5**
- Shallow (10–70 km): **×1.0**
- Intermediate (70–300 km): **×0.6**
- Deep (> 300 km): **×0.2**

### Step 2 — Weighted Average

$$R_{avg} = w_f \cdot S_{flood} + w_e \cdot S_{earthquake} + w_c \cdot S_{cyclone}$$

| Weight | Default | Rationale |
|--------|---------|-----------|
| $w_f$  | **0.40** | Floods are the most frequent natural disaster globally |
| $w_e$  | **0.30** | Earthquakes: rare but devastating, no prediction possible |
| $w_c$  | **0.30** | Cyclones: seasonal, high-impact, hours of lead time |

Weights are configurable per-region (coastal: raise $w_c$; seismic zone: raise $w_e$).

### Step 3 — Max Component

$$R_{max} = \max(S_{flood},\; S_{earthquake},\; S_{cyclone})$$

This prevents a single catastrophic hazard from being diluted by calm values of the other two.

### Step 4 — Hybrid Blending

$$R_{hybrid} = \beta \cdot R_{max} + (1 - \beta) \cdot R_{avg}$$

Default **β = 0.60** (max-dominated). A pure average (β = 0) would rate "M8 earthquake + no flood + no cyclone" as only 30%, which is dangerously misleading.

### Step 5 — Concurrency Amplification

When multiple hazards are simultaneously active (normalised score ≥ 0.30):

$$n_{active} = \text{count of hazards with } S \geq 0.30$$

$$\text{amplifier} = 1 + \gamma \cdot (n_{active} - 1), \quad \gamma = 0.10$$

| Active Hazards | Amplifier |
|:--------------:|:---------:|
| 0 or 1         | ×1.00     |
| 2              | ×1.10     |
| 3              | ×1.20     |

This reflects real-world compounding: earthquake + heavy rain → landslides.

### Step 6 — Final Score

$$\text{overall\_risk\_score} = \text{clamp}(R_{hybrid} \times \text{amplifier} \times 100,\; 0,\; 100)$$

---

## Escalation Thresholds

| Score Range | Level       | Action                | UI Colour | Icon         |
|:-----------:|:-----------:|-----------------------|-----------|--------------|
| 0 – 20     | **Safe**    | Monitor only          | `#4CAF50` | ✅ check      |
| 20 – 45    | **Watch**   | Stay informed         | `#FF9800` | 👁 visibility |
| 45 – 70    | **Warning** | Prepare; secure property | `#F44336` | ⚠ warning   |
| 70 – 100   | **Severe**  | Evacuate / shelter NOW | `#B71C1C` | 🚨 emergency |

### Hysteresis (Anti-Oscillation)

To prevent flapping between levels at boundaries:

- **Escalation** is immediate (score crosses upper threshold → upgrade)
- **De-escalation** requires the score to drop an extra **7 points** below the threshold

| Transition             | Escalation At | De-escalation At |
|:-----------------------|:-------------:|:----------------:|
| Safe → Watch           | ≥ 20          | ≤ 13             |
| Watch → Warning        | ≥ 45          | ≤ 38             |
| Warning → Severe       | ≥ 70          | ≤ 63             |

---

## Alert Triggering Logic

An alert fires when **any** of the following conditions is met:

### Trigger 1 — Level Escalation
The `overall_risk_level` moves to a higher tier compared to the previous assessment.

### Trigger 2 — Individual Hazard Critical
Any single hazard exceeds its own critical threshold:

| Hazard     | Critical Threshold |
|------------|:------------------:|
| Flood      | ≥ 0.80 probability |
| Earthquake | ≥ 0.80 normalised (≈ M7+ shallow) |
| Cyclone    | ≥ 0.80 composite   |

### Trigger 3 — Concurrent Active Hazards
Two or more hazards simultaneously have normalised scores ≥ 0.30.

---

## Priority Ordering of Disasters

When multiple alerts fire simultaneously, they are ranked by **immediacy × lethality**:

| Priority | Disaster     | Reasoning |
|:--------:|:-------------|-----------|
| **1**    | Earthquake   | Zero warning time (seconds at best). Structural collapse is immediately lethal. Triggers secondary hazards (tsunami, landslide). |
| **2**    | Cyclone      | Hours of lead time via satellite. Large area of impact (100s km). Storm surge + wind + flooding combined. |
| **3**    | Flood        | Minutes to hours of lead time. Usually localised. Progressive onset allows staged evacuation. |

The hazard breakdown in the API response is always sorted by this priority order.

---

## API Endpoints

### `POST /api/v1/risk/aggregate`

Compute the unified risk score.

**Request Body:**
```json
{
  "latitude": 13.08,
  "longitude": 80.27,
  "flood_probability": 0.65,
  "earthquake_magnitude": 5.5,
  "earthquake_depth_km": 15.0,
  "cyclone_score": 0.45,
  "previous_level": "watch"
}
```

**Response:**
```json
{
  "overall_risk_score": 58.42,
  "overall_risk_score_pct": "58.4%",
  "overall_risk_level": "warning",
  "alert_action": "prepare",
  "dominant_hazard": "earthquake",
  "active_hazard_count": 3,
  "alert_triggered": true,
  "alert_reasons": [
    "Multiple concurrent hazards active: flood, earthquake, cyclone (3 hazards ≥ 0.3)"
  ],
  "alert_info": {
    "title": "Risk Warning",
    "message": "Significant risk! Prepare emergency supplies...",
    "color": "#F44336",
    "icon": "warning"
  },
  "hazard_breakdown": [
    {
      "hazard_type": "earthquake",
      "raw_value": 5.5,
      "normalised_score": 0.55,
      "weight": 0.30,
      "weighted_contribution": 0.165,
      "is_active": true,
      "is_critical": false,
      "priority": 1
    },
    { "hazard_type": "cyclone", "..." : "..." },
    { "hazard_type": "flood", "..." : "..." }
  ],
  "formula_components": {
    "R_avg": 0.3145,
    "R_max": 0.65,
    "beta": 0.60,
    "R_hybrid": 0.5158,
    "amplifier": 1.20,
    "weights": { "flood": 0.40, "earthquake": 0.30, "cyclone": 0.30 }
  }
}
```

### `GET /api/v1/risk/thresholds`

Returns all tuning parameters, escalation thresholds, and priority configuration.

### `GET /api/v1/risk/health`

Health check for the risk aggregation module.

---

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ flood_service │    │ earthquake_service│    │ cyclone_service  │
│  P ∈ [0,1]   │    │  M, depth_km     │    │  composite [0,1] │
└──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘
       │                     │                       │
       ▼                     ▼                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │              risk_aggregator.aggregate_risk()            │
  │                                                         │
  │  1. Normalise each hazard → S ∈ [0,1]                  │
  │  2. R_avg = Σ(w·S)                                     │
  │  3. R_max = max(S)                                     │
  │  4. R_hybrid = β·R_max + (1−β)·R_avg                   │
  │  5. Amplify for concurrent hazards                      │
  │  6. Classify level + trigger alerts                     │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   AggregatedRisk      │
              │  score: 0–100%        │
              │  level: Safe/Watch/   │
              │         Warning/Severe│
              │  alert_triggered: bool│
              │  dominant_hazard: str │
              └───────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `backend/app/ml/risk_aggregator.py` | Core engine — normalisation, formula, classification, alerts |
| `backend/app/api/v1/risk.py` | FastAPI endpoint (`POST /api/v1/risk/aggregate`) |
| `tests/test_risk_aggregator.py` | 46 tests covering all formula components and edge cases |
| `docs/RISK_AGGREGATION.md` | This document |
