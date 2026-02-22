"use client";

import { type ForecastHorizon } from "@/lib/api";

interface ForecastTimeSliderProps {
  value: ForecastHorizon;
  onChange: (horizon: ForecastHorizon) => void;
}

const HORIZONS: ForecastHorizon[] = [
  { minutes: 10,  label: "10 min" },
  { minutes: 20,  label: "20 min" },
  { minutes: 30,  label: "30 min" },
  { minutes: 45,  label: "45 min" },
  { minutes: 60,  label: "1 hr" },
  { minutes: 120, label: "2 hr" },
  { minutes: 180, label: "3 hr" },
  { minutes: 360, label: "6 hr" },
];

export function ForecastTimeSlider({ value, onChange }: ForecastTimeSliderProps) {
  const activeIdx = HORIZONS.findIndex((h) => h.minutes === value.minutes);

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Forecast Horizon
      </h3>

      {/* Current value */}
      <div className="text-center">
        <span className="text-2xl font-bold text-white">{value.label}</span>
        <p className="text-[10px] text-gray-500 mt-0.5">ahead</p>
      </div>

      {/* Slider track */}
      <div className="relative pt-2 pb-4">
        <input
          type="range"
          min={0}
          max={HORIZONS.length - 1}
          value={activeIdx >= 0 ? activeIdx : 2}
          onChange={(e) => onChange(HORIZONS[parseInt(e.target.value)])}
          className="w-full h-1 bg-surface-3 rounded-full appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-accent-blue
            [&::-webkit-slider-thumb]:shadow-lg
            [&::-webkit-slider-thumb]:cursor-pointer"
        />

        {/* Tick labels */}
        <div className="flex justify-between mt-1">
          {HORIZONS.map((h, i) => (
            <span
              key={h.minutes}
              className={`text-[9px] ${
                i === activeIdx ? "text-accent-blue font-medium" : "text-gray-600"
              }`}
            >
              {h.label}
            </span>
          ))}
        </div>
      </div>

      {/* Quick presets */}
      <div className="grid grid-cols-4 gap-1">
        {[
          { minutes: 30, label: "30m" },
          { minutes: 60, label: "1h" },
          { minutes: 180, label: "3h" },
          { minutes: 360, label: "6h" },
        ].map((h) => (
          <button
            key={h.minutes}
            onClick={() =>
              onChange(HORIZONS.find((x) => x.minutes === h.minutes) || value)
            }
            className={`py-1 rounded text-[10px] font-medium transition-colors ${
              value.minutes === h.minutes
                ? "bg-accent-blue text-white"
                : "bg-surface-2 text-gray-400 hover:text-white"
            }`}
          >
            {h.label}
          </button>
        ))}
      </div>
    </div>
  );
}
