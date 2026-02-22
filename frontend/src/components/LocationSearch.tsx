"use client";

import { useState, useCallback } from "react";

interface LocationSearchProps {
  lat: number;
  lng: number;
  radiusKm: number;
  onLocationChange: (lat: number, lng: number) => void;
  onRadiusChange: (km: number) => void;
}

const RADIUS_PRESETS = [5, 10, 20, 50, 100];

export function LocationSearch({
  lat,
  lng,
  radiusKm,
  onLocationChange,
  onRadiusChange,
}: LocationSearchProps) {
  const [latInput, setLatInput] = useState(lat.toString());
  const [lngInput, setLngInput] = useState(lng.toString());
  const [gpsLoading, setGpsLoading] = useState(false);

  const handleGPS = useCallback(() => {
    if (!navigator.geolocation) return;
    setGpsLoading(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        setLatInput(latitude.toFixed(4));
        setLngInput(longitude.toFixed(4));
        onLocationChange(latitude, longitude);
        setGpsLoading(false);
      },
      () => setGpsLoading(false),
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }, [onLocationChange]);

  const handleSubmit = () => {
    const parsedLat = parseFloat(latInput);
    const parsedLng = parseFloat(lngInput);
    if (!isNaN(parsedLat) && !isNaN(parsedLng)) {
      onLocationChange(parsedLat, parsedLng);
    }
  };

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Location
      </h3>

      {/* Lat/Lon inputs */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-[10px] text-gray-500 mb-1 block">Latitude</label>
          <input
            type="number"
            step="0.0001"
            value={latInput}
            onChange={(e) => setLatInput(e.target.value)}
            onBlur={handleSubmit}
            className="w-full px-2 py-1.5 bg-surface-2 border border-surface-3 rounded-lg text-sm font-mono text-gray-200 focus:border-accent-blue focus:outline-none transition-colors"
          />
        </div>
        <div>
          <label className="text-[10px] text-gray-500 mb-1 block">Longitude</label>
          <input
            type="number"
            step="0.0001"
            value={lngInput}
            onChange={(e) => setLngInput(e.target.value)}
            onBlur={handleSubmit}
            className="w-full px-2 py-1.5 bg-surface-2 border border-surface-3 rounded-lg text-sm font-mono text-gray-200 focus:border-accent-blue focus:outline-none transition-colors"
          />
        </div>
      </div>

      {/* GPS Button */}
      <button
        onClick={handleGPS}
        disabled={gpsLoading}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-surface-2 border border-surface-3 rounded-lg text-sm text-gray-300 hover:text-white hover:border-accent-blue transition-colors disabled:opacity-50"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        {gpsLoading ? "Locating..." : "Use GPS Location"}
      </button>

      {/* Radius presets */}
      <div>
        <label className="text-[10px] text-gray-500 mb-1.5 block">
          Radius: {radiusKm} km
        </label>
        <div className="flex gap-1.5">
          {RADIUS_PRESETS.map((r) => (
            <button
              key={r}
              onClick={() => onRadiusChange(r)}
              className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                radiusKm === r
                  ? "bg-accent-blue text-white"
                  : "bg-surface-2 text-gray-400 hover:text-white"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
