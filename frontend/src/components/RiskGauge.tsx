"use client";

import { useState } from "react";
import clsx from "clsx";

interface RiskGaugeProps {
  score: number;     // 0–100
  level: string;     // safe | watch | warning | severe
  dominantHazard: string;
  loading?: boolean;
  hazardBreakdown?: { hazard_type: string; normalised_score: number }[];
  onHazardSelect?: (hazard: string) => void;
}

const LEVEL_COLORS: Record<string, string> = {
  safe:    "#10B981",
  watch:   "#F59E0B",
  warning: "#F97316",
  severe:  "#EF4444",
};

const HAZARD_COLORS: Record<string, string> = {
  flood:      "#3B82F6",
  earthquake: "#EF4444",
  cyclone:    "#8B5CF6",
};

const HAZARD_ICONS: Record<string, string> = {
  flood:      "🌊",
  earthquake: "🌍",
  cyclone:    "🌀",
};

const LEVEL_LABELS: Record<string, string> = {
  safe:    "Safe",
  watch:   "Watch",
  warning: "Warning",
  severe:  "Severe",
};

/**
 * Semi-circular SVG risk gauge (0–100%).
 *
 * Arc geometry:
 *   - Centre at (100, 100), radius 80
 *   - Arc from 180° to 0° (left to right, bottom half of circle)
 *   - Circumference of semicircle = π × r = 251.33
 *   - dashoffset = circumference × (1 - score/100)
 */
export function RiskGauge({ score, level, dominantHazard, loading, hazardBreakdown, onHazardSelect }: RiskGaugeProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedHazard, setSelectedHazard] = useState<string | null>(null);
  
  const color = LEVEL_COLORS[level] || LEVEL_COLORS.safe;
  const circumference = Math.PI * 80; // ≈ 251.33
  
  // Get the score for the selected hazard or overall
  const displayHazard = selectedHazard || dominantHazard;
  const hazardData = hazardBreakdown?.find(h => h.hazard_type === displayHazard);
  const displayScore = selectedHazard && hazardData 
    ? hazardData.normalised_score * 100 
    : score;
  
  const displayColor = selectedHazard 
    ? (HAZARD_COLORS[selectedHazard] || color)
    : color;
  
  const offset = circumference * (1 - Math.min(displayScore, 100) / 100);

  const handleHazardClick = () => {
    setShowDropdown(!showDropdown);
  };

  const selectHazard = (hazard: string | null) => {
    setSelectedHazard(hazard);
    setShowDropdown(false);
    if (onHazardSelect && hazard) {
      onHazardSelect(hazard);
    }
  };

  const hazardTypes = ["flood", "earthquake", "cyclone"];

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        {selectedHazard ? `${selectedHazard.toUpperCase()} RISK` : "OVERALL RISK"}
      </h3>

      <div className="relative flex flex-col items-center">
        {/* SVG Gauge */}
        <svg viewBox="0 0 200 120" className="w-full max-w-[220px]">
          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#242430"
            strokeWidth="12"
            strokeLinecap="round"
          />

          {/* Active arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={displayColor}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="risk-gauge-arc"
            style={{
              filter: displayScore >= 70 ? `drop-shadow(0 0 8px ${displayColor})` : undefined,
              transition: "stroke-dashoffset 0.5s ease, stroke 0.3s ease",
            }}
          />

          {/* Score text */}
          <text
            x="100"
            y="85"
            textAnchor="middle"
            className="fill-white text-3xl font-bold"
            style={{ fontSize: "36px", fontFamily: "Inter, sans-serif" }}
          >
            {loading ? "..." : Math.round(displayScore)}
          </text>
          <text
            x="100"
            y="105"
            textAnchor="middle"
            className="fill-gray-400 text-xs"
            style={{ fontSize: "12px", fontFamily: "Inter, sans-serif" }}
          >
            {loading ? "Loading" : selectedHazard ? `${Math.round(displayScore)}% Risk` : `${LEVEL_LABELS[level] || level}`}
          </text>
        </svg>

        {/* Clickable Hazard badge with dropdown */}
        {!loading && (
          <div className="relative mt-1">
            <button
              onClick={handleHazardClick}
              className={clsx(
                "px-3 py-1.5 rounded-full text-xs font-semibold capitalize",
                "border cursor-pointer transition-all duration-200",
                "hover:scale-105 hover:shadow-lg",
                "flex items-center gap-1.5"
              )}
              style={{
                borderColor: displayColor,
                color: displayColor,
                backgroundColor: `${displayColor}20`,
              }}
            >
              <span>{HAZARD_ICONS[displayHazard] || "⚠️"}</span>
              <span>{selectedHazard ? displayHazard : `Dominant: ${dominantHazard}`}</span>
              <svg 
                className={clsx("w-3 h-3 transition-transform", showDropdown && "rotate-180")} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {/* Dropdown menu */}
            {showDropdown && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 bg-surface-2 border border-surface-3 rounded-lg shadow-xl z-50 overflow-hidden min-w-[160px]">
                {/* Overall option */}
                <button
                  onClick={() => selectHazard(null)}
                  className={clsx(
                    "w-full px-4 py-2.5 text-left text-sm flex items-center gap-2",
                    "hover:bg-surface-3 transition-colors",
                    !selectedHazard && "bg-surface-3"
                  )}
                >
                  <span>📊</span>
                  <span className="text-gray-200">Overall Risk</span>
                  {!selectedHazard && <span className="ml-auto text-green-400">✓</span>}
                </button>
                
                <div className="border-t border-surface-3" />
                
                {/* Hazard options */}
                {hazardTypes.map((hazard) => {
                  const hData = hazardBreakdown?.find(h => h.hazard_type === hazard);
                  const hScore = hData ? Math.round(hData.normalised_score * 100) : 0;
                  const hColor = HAZARD_COLORS[hazard];
                  
                  return (
                    <button
                      key={hazard}
                      onClick={() => selectHazard(hazard)}
                      className={clsx(
                        "w-full px-4 py-2.5 text-left text-sm flex items-center gap-2",
                        "hover:bg-surface-3 transition-colors",
                        selectedHazard === hazard && "bg-surface-3"
                      )}
                    >
                      <span>{HAZARD_ICONS[hazard]}</span>
                      <span className="capitalize text-gray-200">{hazard}</span>
                      <span 
                        className="ml-auto text-xs font-mono px-1.5 py-0.5 rounded"
                        style={{ backgroundColor: `${hColor}30`, color: hColor }}
                      >
                        {hScore}%
                      </span>
                      {selectedHazard === hazard && <span className="text-green-400">✓</span>}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
