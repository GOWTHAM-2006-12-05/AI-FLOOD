"use client";

import { useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import { AlertBanner } from "@/components/AlertBanner";
import { Sidebar } from "@/components/Sidebar";
import { RiskGauge } from "@/components/RiskGauge";
import { FeatureImportancePanel } from "@/components/FeatureImportancePanel";
import { ModelConfidencePanel } from "@/components/ModelConfidencePanel";
import { ForecastTimeSlider } from "@/components/ForecastTimeSlider";
import { LocationSearch } from "@/components/LocationSearch";
import { apiClient, type RiskData, type ForecastHorizon } from "@/lib/api";

// Leaflet must be loaded client-side only (no SSR)
const DisasterMap = dynamic(() => import("@/components/DisasterMap"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-surface-1 rounded-xl">
      <div className="text-gray-400 animate-pulse">Loading map...</div>
    </div>
  ),
});

export default function DashboardPage() {
  // ── State ──
  const [location, setLocation] = useState<{ lat: number; lng: number }>({
    lat: 13.0827,
    lng: 80.2707,
  });
  const [radiusKm, setRadiusKm] = useState(50);
  const [riskData, setRiskData] = useState<RiskData | null>(null);
  const [forecastHorizon, setForecastHorizon] = useState<ForecastHorizon>({
    minutes: 30,
    label: "30 min",
  });
  const [alerts, setAlerts] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [showGrid, setShowGrid] = useState(true);

  // ── Fetch risk data ──
  const fetchRisk = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiClient.getRiskAssessment(
        location.lat,
        location.lng,
        radiusKm
      );
      setRiskData(data);

      // Extract active alerts
      if (data.alert_reasons?.length) {
        setAlerts(data.alert_reasons);
      }
    } catch (err) {
      console.error("Risk fetch failed:", err);
    } finally {
      setLoading(false);
    }
  }, [location, radiusKm]);

  useEffect(() => {
    fetchRisk();
  }, [fetchRisk]);

  // ── Handlers ──
  const handleLocationChange = (lat: number, lng: number) => {
    setLocation({ lat, lng });
  };

  const handleRadiusChange = (km: number) => {
    setRadiusKm(km);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ── Alert Banner (top) ── */}
      {alerts.length > 0 && riskData && (
        <AlertBanner
          alerts={alerts}
          riskLevel={riskData.overall_risk_level}
          onDismiss={() => setAlerts([])}
        />
      )}

      {/* ── Main Layout ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* ── Left Sidebar ── */}
        <Sidebar className="w-80 flex-shrink-0">
          <LocationSearch
            lat={location.lat}
            lng={location.lng}
            radiusKm={radiusKm}
            onLocationChange={handleLocationChange}
            onRadiusChange={handleRadiusChange}
          />

          <RiskGauge
            score={riskData?.overall_risk_score ?? 0}
            level={riskData?.overall_risk_level ?? "safe"}
            dominantHazard={riskData?.dominant_hazard ?? "none"}
            loading={loading}
            hazardBreakdown={riskData?.hazard_breakdown}
          />

          <ModelConfidencePanel
            confidence={riskData?.model_confidence}
            ensembleAlpha={riskData?.ensemble_alpha ?? 0.65}
            xgbWeight={0.65}
            lstmWeight={0.35}
          />

          <FeatureImportancePanel
            features={riskData?.feature_importance}
          />
        </Sidebar>

        {/* ── Map (centre, fills remaining space) ── */}
        <main className="flex-1 relative">
          <DisasterMap
            center={[location.lat, location.lng]}
            radiusKm={radiusKm}
            riskData={riskData}
            showGrid={showGrid}
            forecastMinutes={forecastHorizon.minutes}
          />

          {/* ── Map overlay controls ── */}
          <div className="absolute top-4 right-4 z-[1000] flex flex-col gap-2">
            <button
              onClick={() => setShowGrid(!showGrid)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                showGrid
                  ? "bg-accent-blue text-white"
                  : "bg-surface-2 text-gray-400 hover:text-white"
              }`}
            >
              Grid Overlay
            </button>
            <button
              onClick={fetchRisk}
              disabled={loading}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-surface-2 text-gray-400 hover:text-white transition-colors disabled:opacity-50"
            >
              {loading ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        </main>

        {/* ── Right Panel ── */}
        <aside className="w-72 flex-shrink-0 bg-surface-1 border-l border-surface-3 p-4 overflow-y-auto">
          <ForecastTimeSlider
            value={forecastHorizon}
            onChange={setForecastHorizon}
          />

          {/* ── Hazard breakdown cards ── */}
          {riskData?.hazard_breakdown && (
            <div className="mt-6 space-y-3">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Hazard Breakdown
              </h3>
              {riskData.hazard_breakdown.map((h) => (
                <div
                  key={h.hazard_type}
                  className="bg-surface-2 rounded-lg p-3 border border-surface-3"
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium capitalize">
                      {h.hazard_type}
                    </span>
                    <span className="text-xs font-mono text-gray-400">
                      {(h.normalised_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-surface-3 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${h.normalised_score * 100}%`,
                        backgroundColor: h.is_critical
                          ? "#EF4444"
                          : h.normalised_score > 0.5
                          ? "#F97316"
                          : "#10B981",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
