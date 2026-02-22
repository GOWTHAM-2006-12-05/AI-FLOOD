"use client";

import type { ModelConfidence } from "@/lib/api";

interface ModelConfidencePanelProps {
  confidence?: ModelConfidence;
  ensembleAlpha: number;
  xgbWeight: number;
  lstmWeight: number;
}

export function ModelConfidencePanel({
  confidence,
  ensembleAlpha,
  xgbWeight,
  lstmWeight,
}: ModelConfidencePanelProps) {
  const xgb = confidence?.xgboost ?? 0;
  const lstm = confidence?.lstm ?? 0;
  const ensemble = confidence?.ensemble ?? 0;
  const agreement = confidence?.agreement ?? true;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Model Confidence
      </h3>

      <div className="bg-surface-2 rounded-lg p-3 border border-surface-3 space-y-3">
        {/* Ensemble formula */}
        <div className="text-[10px] text-gray-500 font-mono text-center">
          P<sub>ensemble</sub> = {ensembleAlpha} × P<sub>xgb</sub> + {(1 - ensembleAlpha).toFixed(2)} × P<sub>lstm</sub>
        </div>

        {/* Model bars */}
        <div className="space-y-2">
          <ModelBar label="XGBoost" value={xgb} weight={xgbWeight} color="#3B82F6" />
          <ModelBar label="LSTM" value={lstm} weight={lstmWeight} color="#8B5CF6" />

          <div className="pt-1 border-t border-surface-3">
            <ModelBar
              label="Ensemble"
              value={ensemble}
              weight={1.0}
              color="#06B6D4"
              highlight
            />
          </div>
        </div>

        {/* Agreement indicator */}
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-gray-500">Model Agreement</span>
          <span
            className={
              agreement ? "text-green-400 font-medium" : "text-amber-400 font-medium"
            }
          >
            {agreement ? "✓ Aligned" : "⚠ Disagreement"}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Sub-component ──

function ModelBar({
  label,
  value,
  weight,
  color,
  highlight = false,
}: {
  label: string;
  value: number;
  weight: number;
  color: string;
  highlight?: boolean;
}) {
  const pct = Math.min(value * 100, 100);

  return (
    <div>
      <div className="flex justify-between items-center mb-0.5">
        <span className="text-[11px] text-gray-300">
          {label}
          <span className="text-gray-600 ml-1">
            ({(weight * 100).toFixed(0)}%)
          </span>
        </span>
        <span className="text-[11px] font-mono text-gray-400">
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="w-full h-1.5 bg-surface-3 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            backgroundColor: color,
            boxShadow: highlight ? `0 0 6px ${color}` : undefined,
          }}
        />
      </div>
    </div>
  );
}
