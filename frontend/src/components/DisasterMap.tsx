"use client";

import { useEffect, useMemo, useRef } from "react";
import {
  MapContainer,
  TileLayer,
  Circle,
  Marker,
  Popup,
  Rectangle,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import type { RiskData, GridCell } from "@/lib/api";

// ── Fix default marker icons (Leaflet + bundlers) ──
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

// ── Props ──
interface DisasterMapProps {
  center: [number, number];
  radiusKm: number;
  riskData: RiskData | null;
  showGrid: boolean;
  forecastMinutes: number;
}

// ── Recenter helper ──
function RecenterMap({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, map.getZoom(), { duration: 0.8 });
  }, [center, map]);
  return null;
}

// ── Risk color helpers ──
function riskColor(score: number): string {
  if (score >= 0.75) return "#EF4444"; // severe
  if (score >= 0.5) return "#F97316"; // warning
  if (score >= 0.25) return "#EAB308"; // watch
  return "#10B981"; // safe
}

function riskFill(score: number): string {
  if (score >= 0.75) return "rgba(239,68,68,0.35)";
  if (score >= 0.5) return "rgba(249,115,22,0.25)";
  if (score >= 0.25) return "rgba(234,179,8,0.18)";
  return "rgba(16,185,129,0.12)";
}

// ── Generate a mock 1-km grid inside the radius ──
function generateGrid(
  center: [number, number],
  radiusKm: number,
  riskScore: number
): GridCell[] {
  const cells: GridCell[] = [];
  const KM_PER_DEG_LAT = 111.32;
  const KM_PER_DEG_LNG = 111.32 * Math.cos((center[0] * Math.PI) / 180);
  const gridSpacingKm = radiusKm <= 10 ? 1 : radiusKm <= 25 ? 2 : 5;
  const halfSteps = Math.floor(radiusKm / gridSpacingKm);

  for (let r = -halfSteps; r <= halfSteps; r++) {
    for (let c = -halfSteps; c <= halfSteps; c++) {
      const dLat = (r * gridSpacingKm) / KM_PER_DEG_LAT;
      const dLng = (c * gridSpacingKm) / KM_PER_DEG_LNG;
      const lat = center[0] + dLat;
      const lng = center[1] + dLng;

      // Check if inside radius
      const distKm = Math.sqrt(
        (dLat * KM_PER_DEG_LAT) ** 2 + (dLng * KM_PER_DEG_LNG) ** 2
      );
      if (distKm > radiusKm) continue;

      // Spatial variation: higher risk near centre
      const distFraction = distKm / radiusKm;
      const spatialFactor = 1 - distFraction * 0.6;
      const noise = 0.85 + Math.random() * 0.3;
      const cellRisk = Math.min(
        1,
        Math.max(0, riskScore * spatialFactor * noise)
      );

      cells.push({
        row: r + halfSteps,
        col: c + halfSteps,
        lat,
        lng,
        flood_risk: cellRisk,
        elevation: 10 + Math.random() * 100,
        drainage: 0.3 + Math.random() * 0.7,
      });
    }
  }
  return cells;
}

// ── Grid overlay sub-component ──
function GridOverlay({
  center,
  radiusKm,
  riskScore,
}: {
  center: [number, number];
  radiusKm: number;
  riskScore: number;
}) {
  const cells = useMemo(
    () => generateGrid(center, radiusKm, riskScore),
    [center, radiusKm, riskScore]
  );

  const KM_PER_DEG_LAT = 111.32;
  const KM_PER_DEG_LNG = 111.32 * Math.cos((center[0] * Math.PI) / 180);
  const gridSpacingKm = radiusKm <= 10 ? 1 : radiusKm <= 25 ? 2 : 5;
  const halfLat = gridSpacingKm / KM_PER_DEG_LAT / 2;
  const halfLng = gridSpacingKm / KM_PER_DEG_LNG / 2;

  return (
    <>
      {cells.map((cell) => {
        const bounds: L.LatLngBoundsExpression = [
          [cell.lat - halfLat, cell.lng - halfLng],
          [cell.lat + halfLat, cell.lng + halfLng],
        ];
        return (
          <Rectangle
            key={`${cell.row}-${cell.col}`}
            bounds={bounds}
            pathOptions={{
              color: riskColor(cell.flood_risk),
              weight: 0.5,
              fillColor: riskFill(cell.flood_risk),
              fillOpacity: 0.6,
            }}
          >
            <Popup>
              <div className="text-xs space-y-1">
                <div className="font-semibold">
                  Grid [{cell.row},{cell.col}]
                </div>
                <div>
                  Flood Risk:{" "}
                  <span className="font-mono">
                    {(cell.flood_risk * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  Elevation:{" "}
                  <span className="font-mono">{cell.elevation.toFixed(0)}m</span>
                </div>
                <div>
                  Drainage:{" "}
                  <span className="font-mono">
                    {(cell.drainage * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="text-gray-500">
                  {cell.lat.toFixed(4)}, {cell.lng.toFixed(4)}
                </div>
              </div>
            </Popup>
          </Rectangle>
        );
      })}
    </>
  );
}

// ── Main Map Component ──
export default function DisasterMap({
  center,
  radiusKm,
  riskData,
  showGrid,
  forecastMinutes,
}: DisasterMapProps) {
  const overallScore = riskData?.overall_risk_score ?? 0;
  const overallLevel = riskData?.overall_risk_level ?? "safe";

  const circleColor = {
    safe: "#10B981",
    watch: "#EAB308",
    warning: "#F97316",
    severe: "#EF4444",
  }[overallLevel];

  return (
    <MapContainer
      center={center}
      zoom={radiusKm <= 10 ? 13 : radiusKm <= 25 ? 11 : radiusKm <= 50 ? 10 : 9}
      className="h-full w-full rounded-xl z-0"
      zoomControl={false}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <RecenterMap center={center} />

      {/* ── Centre marker ── */}
      <Marker position={center}>
        <Popup>
          <div className="text-xs space-y-1">
            <div className="font-semibold">Monitoring Centre</div>
            <div>
              {center[0].toFixed(4)}, {center[1].toFixed(4)}
            </div>
            <div>
              Radius: <span className="font-mono">{radiusKm} km</span>
            </div>
            {riskData && (
              <>
                <div>
                  Risk:{" "}
                  <span className="font-mono">
                    {(overallScore * 100).toFixed(1)}%
                  </span>{" "}
                  ({overallLevel})
                </div>
                <div>Dominant: {riskData.dominant_hazard}</div>
                <div>Forecast: {forecastMinutes} min</div>
              </>
            )}
          </div>
        </Popup>
      </Marker>

      {/* ── Monitoring radius circle ── */}
      <Circle
        center={center}
        radius={radiusKm * 1000}
        pathOptions={{
          color: circleColor,
          weight: 2,
          fillColor: circleColor,
          fillOpacity: 0.08,
          dashArray: "8 4",
        }}
      />

      {/* ── Hyper-local grid overlay ── */}
      {showGrid && (
        <GridOverlay
          center={center}
          radiusKm={radiusKm}
          riskScore={overallScore}
        />
      )}
    </MapContainer>
  );
}
