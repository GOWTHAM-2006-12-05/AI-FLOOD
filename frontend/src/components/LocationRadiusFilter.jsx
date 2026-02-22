/**
 * LocationRadiusFilter.jsx
 *
 * Frontend component for the disaster alert radius filter.
 * Handles:
 *   - Browser geolocation (GPS)
 *   - Manual coordinate entry
 *   - Radius selector (5 / 10 / 20 / 50 km)
 *   - API call to POST /api/v1/disasters/nearby
 *   - Display of filtered results
 *
 * Dependencies: React 18+, fetch API (no axios needed)
 */

import React, { useState, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const RADIUS_OPTIONS = [
  { value: 5,  label: "5 km",  description: "Immediate vicinity" },
  { value: 10, label: "10 km", description: "City zone" },
  { value: 20, label: "20 km", description: "District area" },
  { value: 50, label: "50 km", description: "Regional coverage" },
];

const SEVERITY_COLORS = {
  1: "#22c55e", // green
  2: "#eab308", // yellow
  3: "#f97316", // orange
  4: "#ef4444", // red
  5: "#7c2d12", // dark red
};

const SEVERITY_LABELS = {
  1: "Low",
  2: "Moderate",
  3: "High",
  4: "Severe",
  5: "Critical",
};

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function LocationRadiusFilter() {
  // --- State ---
  const [latitude, setLatitude]   = useState("");
  const [longitude, setLongitude] = useState("");
  const [radiusKm, setRadiusKm]   = useState(10);
  const [source, setSource]       = useState("manual"); // "gps" | "manual"
  const [loading, setLoading]     = useState(false);
  const [gpsLoading, setGpsLoading] = useState(false);
  const [error, setError]         = useState(null);
  const [results, setResults]     = useState(null);

  // --- GPS geolocation ---
  const handleUseGPS = useCallback(() => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser.");
      return;
    }

    setGpsLoading(true);
    setError(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLatitude(position.coords.latitude.toFixed(6));
        setLongitude(position.coords.longitude.toFixed(6));
        setSource("gps");
        setGpsLoading(false);
      },
      (err) => {
        const messages = {
          1: "Location permission denied. Please allow access or enter manually.",
          2: "Location unavailable. Try again or enter coordinates manually.",
          3: "Location request timed out. Try again.",
        };
        setError(messages[err.code] || "Failed to get location.");
        setGpsLoading(false);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );
  }, []);

  // --- Manual input handlers ---
  const handleManualChange = useCallback(() => {
    setSource("manual");
  }, []);

  // --- Fetch nearby disasters ---
  const handleSearch = useCallback(async () => {
    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);

    if (isNaN(lat) || lat < -90 || lat > 90) {
      setError("Latitude must be between -90 and 90.");
      return;
    }
    if (isNaN(lon) || lon < -180 || lon > 180) {
      setError("Longitude must be between -180 and 180.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/disasters/nearby`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location: { latitude: lat, longitude: lon, source },
          radius_km: radiusKm,
          min_severity: 1,
          hazard_types: null,
          max_results: 50,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(`Failed to fetch disasters: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [latitude, longitude, radiusKm, source]);

  // --- Render ---
  return (
    <div style={styles.container}>
      <h2 style={styles.title}>🛰️ Disaster Alert — Radius Filter</h2>

      {/* ---- Location Input Section ---- */}
      <div style={styles.section}>
        <h3 style={styles.sectionTitle}>📍 Your Location</h3>

        <button
          onClick={handleUseGPS}
          disabled={gpsLoading}
          style={styles.gpsButton}
        >
          {gpsLoading ? "📡 Getting location..." : "📡 Use My GPS Location"}
        </button>

        <div style={styles.divider}>— or enter manually —</div>

        <div style={styles.inputRow}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Latitude</label>
            <input
              type="number"
              step="0.0001"
              min="-90"
              max="90"
              value={latitude}
              onChange={(e) => { setLatitude(e.target.value); handleManualChange(); }}
              placeholder="e.g. 13.0827"
              style={styles.input}
            />
          </div>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Longitude</label>
            <input
              type="number"
              step="0.0001"
              min="-180"
              max="180"
              value={longitude}
              onChange={(e) => { setLongitude(e.target.value); handleManualChange(); }}
              placeholder="e.g. 80.2707"
              style={styles.input}
            />
          </div>
        </div>

        {source === "gps" && (
          <div style={styles.sourceTag}>✅ Location from GPS</div>
        )}
      </div>

      {/* ---- Radius Selector ---- */}
      <div style={styles.section}>
        <h3 style={styles.sectionTitle}>📏 Alert Radius</h3>
        <div style={styles.radiusRow}>
          {RADIUS_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setRadiusKm(opt.value)}
              style={{
                ...styles.radiusButton,
                ...(radiusKm === opt.value ? styles.radiusActive : {}),
              }}
            >
              <strong>{opt.label}</strong>
              <small style={styles.radiusDesc}>{opt.description}</small>
            </button>
          ))}
        </div>
      </div>

      {/* ---- Search Button ---- */}
      <button
        onClick={handleSearch}
        disabled={loading || !latitude || !longitude}
        style={{
          ...styles.searchButton,
          opacity: loading || !latitude || !longitude ? 0.5 : 1,
        }}
      >
        {loading ? "🔍 Searching..." : "🔍 Find Nearby Disasters"}
      </button>

      {/* ---- Error ---- */}
      {error && <div style={styles.error}>⚠️ {error}</div>}

      {/* ---- Results ---- */}
      {results && (
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>
            🚨 Results — {results.results_count} disaster(s) within{" "}
            {results.radius_km} km
          </h3>
          <p style={styles.meta}>
            Checked {results.total_checked} events · Excluded{" "}
            {results.excluded_count}
          </p>

          {results.disasters.length === 0 ? (
            <div style={styles.noResults}>
              ✅ No active disasters in your area. Stay safe!
            </div>
          ) : (
            <div style={styles.resultsList}>
              {results.disasters.map((d) => (
                <div key={d.id} style={styles.card}>
                  <div style={styles.cardHeader}>
                    <span
                      style={{
                        ...styles.severityBadge,
                        backgroundColor: SEVERITY_COLORS[d.severity],
                      }}
                    >
                      {SEVERITY_LABELS[d.severity]}
                    </span>
                    <span style={styles.hazardTag}>{d.hazard_type}</span>
                    <span style={styles.distance}>{d.distance_display}</span>
                  </div>
                  <h4 style={styles.cardTitle}>{d.title}</h4>
                  <p style={styles.cardDesc}>{d.description}</p>
                  <div style={styles.cardFooter}>
                    <span>📡 {d.source}</span>
                    <span>🕐 {new Date(d.timestamp).toLocaleString()}</span>
                    <span>ID: {d.id}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Styles (inline for portability — extract to CSS/Tailwind in prod)  */
/* ------------------------------------------------------------------ */

const styles = {
  container: {
    maxWidth: 720,
    margin: "2rem auto",
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    padding: "0 1rem",
    color: "#1e293b",
  },
  title: {
    fontSize: "1.5rem",
    marginBottom: "1.5rem",
    borderBottom: "2px solid #e2e8f0",
    paddingBottom: "0.5rem",
  },
  section: {
    marginBottom: "1.5rem",
    padding: "1rem",
    background: "#f8fafc",
    borderRadius: 8,
    border: "1px solid #e2e8f0",
  },
  sectionTitle: {
    fontSize: "1rem",
    marginBottom: "0.75rem",
    color: "#334155",
  },
  gpsButton: {
    width: "100%",
    padding: "0.75rem",
    fontSize: "1rem",
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    marginBottom: "0.75rem",
  },
  divider: {
    textAlign: "center",
    color: "#94a3b8",
    fontSize: "0.85rem",
    margin: "0.5rem 0",
  },
  inputRow: {
    display: "flex",
    gap: "1rem",
  },
  inputGroup: {
    flex: 1,
  },
  label: {
    display: "block",
    fontSize: "0.85rem",
    marginBottom: 4,
    color: "#64748b",
  },
  input: {
    width: "100%",
    padding: "0.5rem",
    fontSize: "1rem",
    border: "1px solid #cbd5e1",
    borderRadius: 4,
    boxSizing: "border-box",
  },
  sourceTag: {
    marginTop: 8,
    fontSize: "0.8rem",
    color: "#16a34a",
  },
  radiusRow: {
    display: "flex",
    gap: "0.5rem",
    flexWrap: "wrap",
  },
  radiusButton: {
    flex: 1,
    minWidth: 100,
    padding: "0.5rem",
    background: "#fff",
    border: "2px solid #e2e8f0",
    borderRadius: 6,
    cursor: "pointer",
    textAlign: "center",
    transition: "all 0.15s",
  },
  radiusActive: {
    borderColor: "#3b82f6",
    background: "#eff6ff",
  },
  radiusDesc: {
    display: "block",
    fontSize: "0.7rem",
    color: "#94a3b8",
    marginTop: 2,
  },
  searchButton: {
    width: "100%",
    padding: "0.85rem",
    fontSize: "1.1rem",
    background: "#dc2626",
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    marginBottom: "1rem",
    fontWeight: "bold",
  },
  error: {
    padding: "0.75rem",
    background: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: 6,
    color: "#b91c1c",
    marginBottom: "1rem",
  },
  meta: {
    fontSize: "0.85rem",
    color: "#64748b",
    marginBottom: "0.75rem",
  },
  noResults: {
    textAlign: "center",
    padding: "2rem",
    color: "#16a34a",
    fontSize: "1.1rem",
  },
  resultsList: {
    display: "flex",
    flexDirection: "column",
    gap: "0.75rem",
  },
  card: {
    padding: "1rem",
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
  },
  cardHeader: {
    display: "flex",
    gap: "0.5rem",
    alignItems: "center",
    marginBottom: "0.5rem",
  },
  severityBadge: {
    color: "#fff",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: "0.75rem",
    fontWeight: "bold",
  },
  hazardTag: {
    background: "#e0e7ff",
    color: "#3730a3",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: "0.75rem",
    textTransform: "uppercase",
  },
  distance: {
    marginLeft: "auto",
    fontSize: "0.85rem",
    fontWeight: "bold",
    color: "#0f766e",
  },
  cardTitle: {
    margin: "0 0 0.25rem",
    fontSize: "1rem",
  },
  cardDesc: {
    fontSize: "0.85rem",
    color: "#475569",
    margin: "0 0 0.5rem",
  },
  cardFooter: {
    display: "flex",
    gap: "1rem",
    fontSize: "0.75rem",
    color: "#94a3b8",
  },
};
