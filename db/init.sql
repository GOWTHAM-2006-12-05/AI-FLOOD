-- ============================================================
-- AI Disaster Prediction — PostgreSQL Schema
-- ============================================================
-- Run automatically by docker-entrypoint-initdb.d on first boot.

-- ── Extensions ──
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";          -- spatial queries

-- ── Weather observations ──
CREATE TABLE IF NOT EXISTS weather_observations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    observed_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    temperature     REAL,
    humidity        REAL,
    wind_speed      REAL,
    wind_direction  REAL,
    pressure        REAL,
    rainfall_1h     REAL DEFAULT 0,
    rainfall_3h     REAL DEFAULT 0,
    rainfall_6h     REAL DEFAULT 0,
    rainfall_24h    REAL DEFAULT 0,
    source          VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_weather_lat_lng   ON weather_observations (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_weather_observed  ON weather_observations (observed_at DESC);

-- ── Risk assessments ──
CREATE TABLE IF NOT EXISTS risk_assessments (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    latitude            DOUBLE PRECISION NOT NULL,
    longitude           DOUBLE PRECISION NOT NULL,
    radius_km           REAL NOT NULL,
    overall_risk_score  REAL NOT NULL,
    overall_risk_level  VARCHAR(16) NOT NULL,
    dominant_hazard     VARCHAR(32),
    flood_score         REAL,
    earthquake_score    REAL,
    cyclone_score       REAL,
    ensemble_alpha      REAL,
    xgb_confidence      REAL,
    lstm_confidence      REAL,
    ensemble_confidence  REAL,
    model_agreement     BOOLEAN,
    alert_action        VARCHAR(32),
    assessed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_risk_location ON risk_assessments (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_risk_assessed ON risk_assessments (assessed_at DESC);

-- ── Alert broadcasts ──
CREATE TABLE IF NOT EXISTS alert_broadcasts (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    risk_assessment_id  UUID REFERENCES risk_assessments(id) ON DELETE SET NULL,
    priority            VARCHAR(16) NOT NULL,
    alert_type          VARCHAR(32) NOT NULL,
    message             TEXT NOT NULL,
    channels            JSONB NOT NULL DEFAULT '[]',
    geo_fence           JSONB,
    total_recipients    INTEGER DEFAULT 0,
    recipients_reached  INTEGER DEFAULT 0,
    status              VARCHAR(16) NOT NULL DEFAULT 'pending',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_alert_status  ON alert_broadcasts (status);
CREATE INDEX IF NOT EXISTS idx_alert_created ON alert_broadcasts (created_at DESC);

-- ── Alert acknowledgements ──
CREATE TABLE IF NOT EXISTS alert_acknowledgements (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id        UUID NOT NULL REFERENCES alert_broadcasts(id) ON DELETE CASCADE,
    recipient_id    VARCHAR(128) NOT NULL,
    channel         VARCHAR(32) NOT NULL,
    acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ack_alert ON alert_acknowledgements (alert_id);

-- ── Grid predictions (hyper-local) ──
CREATE TABLE IF NOT EXISTS grid_predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_id   UUID REFERENCES risk_assessments(id) ON DELETE CASCADE,
    grid_row        INTEGER NOT NULL,
    grid_col        INTEGER NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    flood_risk      REAL NOT NULL,
    elevation       REAL,
    drainage        REAL,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_grid_assessment ON grid_predictions (assessment_id);

-- ── Forecast history ──
CREATE TABLE IF NOT EXISTS forecast_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    predicted_risk  REAL NOT NULL,
    actual_risk     REAL,
    model_version   VARCHAR(32),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_forecast_created ON forecast_history (created_at DESC);
