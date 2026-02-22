"use client";

import type { FeatureImportance } from "@/lib/api";

interface FeatureImportancePanelProps {
  features?: FeatureImportance[];
}

// Default features when no data is available
const DEFAULT_FEATURES: FeatureImportance[] = [
  { feature: "Rain_24hr",         importance: 0.28, category: "rainfall" },
  { feature: "Rain_6hr",          importance: 0.22, category: "rainfall" },
  { feature: "Rain_3hr",          importance: 0.15, category: "rainfall" },
  { feature: "Rain_1hr",          importance: 0.10, category: "rainfall" },
  { feature: "Soil_moisture",     importance: 0.09, category: "terrain" },
  { feature: "Elevation",         importance: 0.07, category: "terrain" },
  { feature: "Drainage_capacity", importance: 0.05, category: "terrain" },
  { feature: "Urbanization",      importance: 0.04, category: "terrain" },
];

const CATEGORY_COLORS: Record<string, string> = {
  rainfall: "#3B82F6",
  terrain:  "#8B5CF6",
  seismic:  "#EF4444",
  wind:     "#F59E0B",
};

export function FeatureImportancePanel({ features }: FeatureImportancePanelProps) {
  const data = features ?? DEFAULT_FEATURES;
  const maxImportance = Math.max(...data.map((f) => f.importance), 0.01);

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Feature Importance
      </h3>

      <div className="space-y-1.5">
        {data.map((f) => {
          const pct = (f.importance / maxImportance) * 100;
          const color = CATEGORY_COLORS[f.category] || "#6B7280";

          return (
            <div key={f.feature} className="group">
              <div className="flex justify-between items-center mb-0.5">
                <span className="text-[11px] text-gray-300 font-mono truncate max-w-[150px]">
                  {f.feature}
                </span>
                <span className="text-[10px] text-gray-500 font-mono">
                  {(f.importance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-1 bg-surface-3 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 pt-1">
        {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
          <div key={cat} className="flex items-center gap-1">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="text-[10px] text-gray-500 capitalize">{cat}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
