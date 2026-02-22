"use client";

import { useState, useEffect } from "react";
import clsx from "clsx";

interface AlertBannerProps {
  alerts: string[];
  riskLevel: "safe" | "watch" | "warning" | "severe";
  onDismiss: () => void;
}

const LEVEL_STYLES: Record<string, { bg: string; border: string; icon: string }> = {
  safe:    { bg: "bg-risk-safe-dark",    border: "border-risk-safe",    icon: "✓" },
  watch:   { bg: "bg-risk-watch-dark",   border: "border-risk-watch",   icon: "⚠" },
  warning: { bg: "bg-risk-warning-dark", border: "border-risk-warning", icon: "🔶" },
  severe:  { bg: "bg-risk-severe-dark",  border: "border-risk-severe",  icon: "🚨" },
};

export function AlertBanner({ alerts, riskLevel, onDismiss }: AlertBannerProps) {
  const style = LEVEL_STYLES[riskLevel] || LEVEL_STYLES.safe;
  const [visible, setVisible] = useState(true);

  // Auto-dismiss safe-level alerts after 10s
  useEffect(() => {
    if (riskLevel === "safe") {
      const timer = setTimeout(() => setVisible(false), 10000);
      return () => clearTimeout(timer);
    }
  }, [riskLevel]);

  if (!visible || alerts.length === 0) return null;

  return (
    <div
      className={clsx(
        "relative flex items-center gap-3 px-4 py-2",
        "border-b animate-slide-down overflow-hidden",
        style.bg,
        style.border
      )}
    >
      {/* ── Icon ── */}
      <span
        className={clsx(
          "flex-shrink-0 text-lg",
          riskLevel === "severe" && "animate-pulse-risk"
        )}
      >
        {style.icon}
      </span>

      {/* ── Scrolling text ── */}
      <div className="flex-1 overflow-hidden whitespace-nowrap">
        <div
          className={clsx(
            "inline-block",
            alerts.length > 1 && "alert-scroll"
          )}
        >
          {alerts.map((alert, i) => (
            <span key={i} className="text-sm mr-12">
              {alert}
              {i < alerts.length - 1 && (
                <span className="mx-4 text-gray-500">•</span>
              )}
            </span>
          ))}
        </div>
      </div>

      {/* ── Risk level badge ── */}
      <span
        className={clsx(
          "px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider",
          riskLevel === "severe" && "bg-red-600 text-white",
          riskLevel === "warning" && "bg-orange-600 text-white",
          riskLevel === "watch" && "bg-amber-600 text-white",
          riskLevel === "safe" && "bg-green-700 text-white"
        )}
      >
        {riskLevel}
      </span>

      {/* ── Dismiss ── */}
      <button
        onClick={() => {
          setVisible(false);
          onDismiss();
        }}
        className="flex-shrink-0 text-gray-400 hover:text-white transition-colors"
        aria-label="Dismiss alert"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}
