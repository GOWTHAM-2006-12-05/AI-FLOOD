"use client";

import React, { useState, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface CalendarDayData {
  date: string;
  temperature_max: number;
  temperature_min: number;
  rain_mm: number;
  weather_description: string;
  flood_probability: number;
  risk_level: string;
  risk_color: string;
  is_forecast: boolean;
  is_today: boolean;
  has_data: boolean;
  contributing_factors: Array<{ factor: string; value: number }>;
}

interface CalendarMonthData {
  year: number;
  month: number;
  latitude: number;
  longitude: number;
  days: CalendarDayData[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
];

const DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

// Dark mode risk colors with glow effects
const RISK_COLORS: Record<string, { bg: string; text: string; border: string; glow: string; gradient: string }> = {
  safe: { 
    bg: "bg-emerald-900/40", 
    text: "text-emerald-400", 
    border: "border-emerald-500/50", 
    glow: "shadow-emerald-500/20",
    gradient: "from-emerald-600 to-emerald-400"
  },
  watch: { 
    bg: "bg-yellow-900/40", 
    text: "text-yellow-400", 
    border: "border-yellow-500/50", 
    glow: "shadow-yellow-500/20",
    gradient: "from-yellow-600 to-yellow-400"
  },
  warning: { 
    bg: "bg-orange-900/40", 
    text: "text-orange-400", 
    border: "border-orange-500/50", 
    glow: "shadow-orange-500/20",
    gradient: "from-orange-600 to-orange-400"
  },
  severe: { 
    bg: "bg-red-900/40", 
    text: "text-red-400", 
    border: "border-red-500/50", 
    glow: "shadow-red-500/20",
    gradient: "from-red-600 to-red-400"
  },
  extreme: { 
    bg: "bg-purple-900/40", 
    text: "text-purple-400", 
    border: "border-purple-500/50", 
    glow: "shadow-purple-500/20",
    gradient: "from-purple-600 to-purple-400"
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// API Functions
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchCalendarMonth(
  year: number,
  month: number,
  lat: number,
  lng: number
): Promise<CalendarMonthData> {
  const params = new URLSearchParams({
    year: year.toString(),
    month: month.toString(),
    lat: lat.toString(),
    lon: lng.toString(),
  });
  
  const res = await fetch(`${API_BASE}/api/v1/calendar/month?${params}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch calendar data: ${res.statusText}`);
  }
  return res.json();
}

async function fetchCalendarDay(
  date: string,
  lat: number,
  lng: number
): Promise<CalendarDayData> {
  const params = new URLSearchParams({
    date,
    lat: lat.toString(),
    lon: lng.toString(),
  });
  
  const res = await fetch(`${API_BASE}/api/v1/calendar/day?${params}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch day data: ${res.statusText}`);
  }
  return res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Day Cell Component
// ─────────────────────────────────────────────────────────────────────────────

interface DayCellProps {
  day: number;
  data: CalendarDayData | null;
  isToday: boolean;
  isSelected: boolean;
  onClick: () => void;
}

function DayCell({ day, data, isToday, isSelected, onClick }: DayCellProps) {
  const defaultStyle = { 
    bg: "bg-slate-800/50", 
    text: "text-slate-400", 
    border: "border-slate-700/50", 
    glow: "",
    gradient: "from-slate-600 to-slate-400"
  };
  const riskStyle = data ? (RISK_COLORS[data.risk_level] || defaultStyle) : defaultStyle;
  
  return (
    <button
      onClick={onClick}
      className={`
        relative p-3 h-28 w-full border rounded-xl transition-all duration-300
        hover:scale-[1.03] hover:shadow-lg cursor-pointer backdrop-blur-sm
        ${riskStyle.bg} ${riskStyle.border}
        ${isSelected ? "ring-2 ring-cyan-400 shadow-lg shadow-cyan-500/30" : ""}
        ${isToday ? "ring-2 ring-cyan-500 shadow-cyan-500/40" : ""}
        group
      `}
    >
      {/* Day number */}
      <div className={`text-sm font-bold ${isToday ? "text-cyan-400" : riskStyle.text}`}>
        {day}
        {isToday && <span className="ml-1 text-xs text-cyan-300">●</span>}
      </div>
      
      {data && (
        <div className="mt-1 space-y-1">
          {/* Temperature range */}
          <div className="flex items-center text-xs font-medium">
            <span className="text-rose-400">↑{data.temperature_max.toFixed(0)}°</span>
            <span className="mx-1 text-slate-500">/</span>
            <span className="text-sky-400">↓{data.temperature_min.toFixed(0)}°</span>
          </div>
          
          {/* Rainfall */}
          {data.rain_mm > 0 && (
            <div className="flex items-center text-xs text-sky-400">
              <span className="mr-1">💧</span>
              <span className="font-medium">{data.rain_mm.toFixed(1)}mm</span>
            </div>
          )}
          
          {/* Flood probability badge */}
          <div className={`
            text-xs font-bold px-2 py-0.5 rounded-full text-center
            ${riskStyle.text} border ${riskStyle.border}
          `}>
            {data.flood_probability.toFixed(0)}%
          </div>
        </div>
      )}
      
      {/* Data source indicator */}
      {data && (
        <div className={`
          absolute top-2 right-2 w-2 h-2 rounded-full
          ${data.is_forecast ? "bg-cyan-400 shadow-lg shadow-cyan-400/50" : "bg-slate-500"}
        `} title={data.is_forecast ? "Forecast" : "Historical"} />
      )}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Day Detail Panel Component (Modal)
// ─────────────────────────────────────────────────────────────────────────────

interface DayDetailPanelProps {
  data: CalendarDayData;
  onClose: () => void;
}

function DayDetailPanel({ data, onClose }: DayDetailPanelProps) {
  const defaultStyle = { bg: "bg-slate-800/50", text: "text-slate-400", border: "border-slate-700", glow: "", gradient: "from-slate-600 to-slate-400" };
  const riskStyle = RISK_COLORS[data.risk_level] || defaultStyle;
  const date = new Date(data.date);
  
  const getRiskEmoji = (level: string) => {
    switch (level) {
      case 'safe': return '✅';
      case 'watch': return '👁️';
      case 'warning': return '⚠️';
      case 'severe': return '🔴';
      case 'extreme': return '🚨';
      default: return '📊';
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div 
        className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl shadow-2xl border border-slate-700/50 p-6 space-y-5 max-w-lg w-full relative overflow-hidden" 
        onClick={(e) => e.stopPropagation()}
      >
        {/* Decorative glow */}
        <div className={`absolute -top-20 -right-20 w-60 h-60 bg-gradient-to-br ${riskStyle.gradient} opacity-20 blur-3xl rounded-full`} />
        
        {/* Header */}
        <div className="flex justify-between items-start relative">
          <div>
            <h3 className="text-2xl font-bold text-white">
              {date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })}
            </h3>
            <p className="text-slate-400 text-sm">{date.getFullYear()}</p>
            <span className={`
              inline-flex items-center gap-1.5 mt-2 px-3 py-1 rounded-full text-sm font-medium
              ${data.is_forecast ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30" : "bg-slate-700/50 text-slate-400 border border-slate-600/50"}
            `}>
              {data.is_forecast ? "📊 Forecast" : "📜 Historical"}
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-2xl leading-none p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
          >
            ✕
          </button>
        </div>
        
        {/* Risk Level Card */}
        <div className={`p-5 rounded-xl ${riskStyle.bg} border ${riskStyle.border} relative overflow-hidden`}>
          <div className={`absolute inset-0 bg-gradient-to-r ${riskStyle.gradient} opacity-10`} />
          <div className="flex justify-between items-center relative">
            <div>
              <div className={`text-xl font-bold uppercase ${riskStyle.text} flex items-center gap-2`}>
                <span className="text-2xl">{getRiskEmoji(data.risk_level)}</span>
                {data.risk_level} Risk
              </div>
              <div className="text-sm text-slate-400 mt-1">Flood Probability</div>
            </div>
            <div className={`text-5xl font-black ${riskStyle.text}`}>
              {data.flood_probability.toFixed(0)}%
            </div>
          </div>
        </div>
        
        {/* Weather Details Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50 hover:border-slate-600/50 transition-colors">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">🌡️ Temperature</div>
            <div className="text-lg font-bold">
              <span className="text-rose-400">{data.temperature_max.toFixed(1)}°</span>
              <span className="text-slate-500 mx-1">/</span>
              <span className="text-sky-400">{data.temperature_min.toFixed(1)}°</span>
            </div>
          </div>
          
          <div className="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50 hover:border-slate-600/50 transition-colors">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">💧 Precipitation</div>
            <div className="text-lg font-bold text-sky-400">
              {data.rain_mm.toFixed(1)} mm
            </div>
          </div>
          
          {data.weather_description && (
            <div className="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50 col-span-2 hover:border-slate-600/50 transition-colors">
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">☁️ Conditions</div>
              <div className="text-lg font-semibold text-slate-200">
                {data.weather_description}
              </div>
            </div>
          )}
          
          {data.contributing_factors && data.contributing_factors.length > 0 && (
            <div className="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50 col-span-2 hover:border-slate-600/50 transition-colors">
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">📊 Contributing Factors</div>
              <div className="space-y-2">
                {data.contributing_factors.map((f, i) => (
                  <div key={i} className="flex justify-between items-center">
                    <span className="text-slate-400 text-sm">{f.factor}</span>
                    <span className="font-bold text-slate-200">{typeof f.value === 'number' ? f.value.toFixed(1) : f.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Risk Legend Component
// ─────────────────────────────────────────────────────────────────────────────

function RiskLegend() {
  const levels = [
    { key: "safe", label: "Safe", emoji: "✅", range: "0-15%" },
    { key: "watch", label: "Watch", emoji: "👁️", range: "15-35%" },
    { key: "warning", label: "Warning", emoji: "⚠️", range: "35-60%" },
    { key: "severe", label: "Severe", emoji: "🔴", range: "60-80%" },
    { key: "extreme", label: "Extreme", emoji: "🚨", range: "80%+" },
  ];
  
  return (
    <div className="bg-slate-800/60 backdrop-blur-sm rounded-xl border border-slate-700/50 p-4">
      <h4 className="text-sm font-bold text-slate-300 mb-3 flex items-center gap-2">
        <span>🌊</span> Flood Risk Legend
      </h4>
      <div className="flex flex-wrap gap-2">
        {levels.map(({ key, label, emoji, range }) => {
          const style = RISK_COLORS[key];
          return (
            <div
              key={key}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${style.bg} ${style.border} hover:scale-105 transition-transform cursor-default`}
              title={range}
            >
              <span>{emoji}</span>
              <span className={`text-xs font-bold ${style.text}`}>{label}</span>
              <span className="text-xs text-slate-500">({range})</span>
            </div>
          );
        })}
      </div>
      <div className="mt-4 flex gap-6 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50" />
          <span>Forecast Data</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-slate-500" />
          <span>Historical Data</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Year Selector Component
// ─────────────────────────────────────────────────────────────────────────────

interface YearSelectorProps {
  year: number;
  onChange: (year: number) => void;
  minYear?: number;
  maxYear?: number;
}

function YearSelector({ year, onChange, minYear = 2000, maxYear }: YearSelectorProps) {
  const currentYear = new Date().getFullYear();
  const max = maxYear ?? currentYear + 1;
  const years = Array.from({ length: max - minYear + 1 }, (_, i) => minYear + i).reverse();
  
  return (
    <select
      value={year}
      onChange={(e) => onChange(parseInt(e.target.value))}
      className="px-4 py-2.5 border border-slate-600 rounded-xl bg-slate-800 text-slate-200 font-medium 
                 focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 cursor-pointer
                 hover:bg-slate-700 transition-colors"
    >
      {years.map((y) => (
        <option key={y} value={y}>
          {y}
        </option>
      ))}
    </select>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Calendar View Component
// ─────────────────────────────────────────────────────────────────────────────

interface CalendarViewProps {
  latitude?: number;
  longitude?: number;
  initialYear?: number;
  initialMonth?: number;
}

export default function CalendarView({
  latitude = 13.0827,  // Default: Chennai
  longitude = 80.2707,
  initialYear,
  initialMonth,
}: CalendarViewProps) {
  const today = new Date();
  const [year, setYear] = useState(initialYear ?? today.getFullYear());
  const [month, setMonth] = useState(initialMonth ?? today.getMonth() + 1);
  const [calendarData, setCalendarData] = useState<CalendarMonthData | null>(null);
  const [selectedDay, setSelectedDay] = useState<CalendarDayData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Map date -> CalendarDayData
  const dayDataMap = new Map<number, CalendarDayData>();
  if (calendarData) {
    for (const d of calendarData.days) {
      const dayNum = new Date(d.date).getDate();
      dayDataMap.set(dayNum, d);
    }
  }
  
  // Calculate calendar grid
  const firstDayOfMonth = new Date(year, month - 1, 1).getDay();
  const daysInMonth = new Date(year, month, 0).getDate();
  
  // Fetch calendar data
  const loadCalendarData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchCalendarMonth(year, month, latitude, longitude);
      setCalendarData(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load calendar data");
    } finally {
      setLoading(false);
    }
  }, [year, month, latitude, longitude]);
  
  useEffect(() => {
    loadCalendarData();
  }, [loadCalendarData]);
  
  // Navigation handlers
  const goToPrevMonth = () => {
    if (month === 1) {
      setYear(year - 1);
      setMonth(12);
    } else {
      setMonth(month - 1);
    }
    setSelectedDay(null);
  };
  
  const goToNextMonth = () => {
    if (month === 12) {
      setYear(year + 1);
      setMonth(1);
    } else {
      setMonth(month + 1);
    }
    setSelectedDay(null);
  };
  
  const goToToday = () => {
    setYear(today.getFullYear());
    setMonth(today.getMonth() + 1);
    setSelectedDay(null);
  };
  
  // Handle day click
  const handleDayClick = async (day: number) => {
    const data = dayDataMap.get(day);
    if (data) {
      setSelectedDay(data);
    } else {
      const dateStr = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
      try {
        const dayData = await fetchCalendarDay(dateStr, latitude, longitude);
        setSelectedDay(dayData);
      } catch (e) {
        console.error("Failed to fetch day data:", e);
      }
    }
  };
  
  // Is this day today?
  const isToday = (day: number) => 
    year === today.getFullYear() && month === today.getMonth() + 1 && day === today.getDate();
  
  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-gradient-to-r from-slate-800/80 via-slate-800/60 to-slate-800/80 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-5 shadow-xl">
        <div className="flex items-center gap-4">
          <button
            onClick={goToPrevMonth}
            className="p-3 hover:bg-slate-700/50 rounded-xl transition-all duration-200 text-slate-400 hover:text-white hover:scale-110"
            aria-label="Previous month"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          
          <h2 className="text-2xl font-black text-white min-w-[200px] text-center tracking-tight">
            {MONTH_NAMES[month - 1]} <span className="text-cyan-400">{year}</span>
          </h2>
          
          <button
            onClick={goToNextMonth}
            className="p-3 hover:bg-slate-700/50 rounded-xl transition-all duration-200 text-slate-400 hover:text-white hover:scale-110"
            aria-label="Next month"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={goToToday}
            className="px-5 py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-xl 
                       hover:from-cyan-500 hover:to-blue-500 transition-all duration-200 font-bold
                       shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:scale-105"
          >
            📅 Today
          </button>
          
          <YearSelector year={year} onChange={(y) => { setYear(y); setSelectedDay(null); }} />
          
          <button
            onClick={loadCalendarData}
            disabled={loading}
            className="px-5 py-2.5 bg-slate-700/50 text-slate-300 rounded-xl border border-slate-600 
                       hover:bg-slate-600/50 transition-all duration-200 font-medium disabled:opacity-50
                       hover:text-white hover:scale-105"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Loading...
              </span>
            ) : "🔄 Refresh"}
          </button>
        </div>
      </div>
      
      {/* Location info */}
      <div className="text-sm text-slate-400 px-2 flex items-center gap-2">
        <span className="text-lg">📍</span>
        <span>Location: <span className="text-cyan-400 font-medium">{latitude.toFixed(4)}°N</span>, <span className="text-cyan-400 font-medium">{longitude.toFixed(4)}°E</span></span>
      </div>
      
      {/* Error message */}
      {error && (
        <div className="bg-red-900/30 border border-red-500/30 text-red-400 px-5 py-4 rounded-xl flex items-center gap-3">
          <span className="text-xl">⚠️</span>
          <span>{error}</span>
        </div>
      )}
      
      {/* Calendar Grid */}
      <div className="bg-slate-800/40 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden shadow-2xl">
        {/* Day headers */}
        <div className="grid grid-cols-7 bg-slate-900/60 border-b border-slate-700/50">
          {DAY_NAMES.map((d, i) => (
            <div key={d} className={`p-4 text-center text-sm font-bold tracking-wider
              ${i === 0 ? "text-red-400" : i === 6 ? "text-orange-400" : "text-slate-400"}`}>
              {d}
            </div>
          ))}
        </div>
        
        {/* Calendar cells */}
        <div className="grid grid-cols-7 gap-2 p-3">
          {/* Empty cells before first day */}
          {Array.from({ length: firstDayOfMonth }).map((_, i) => (
            <div key={`empty-${i}`} className="h-28" />
          ))}
          
          {/* Day cells */}
          {Array.from({ length: daysInMonth }, (_, i) => i + 1).map((day) => (
            <DayCell
              key={day}
              day={day}
              data={dayDataMap.get(day) || null}
              isToday={isToday(day)}
              isSelected={selectedDay?.date === `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`}
              onClick={() => handleDayClick(day)}
            />
          ))}
        </div>
      </div>
      
      {/* Risk Legend */}
      <RiskLegend />
      
      {/* Selected Day Detail Panel */}
      {selectedDay && (
        <DayDetailPanel data={selectedDay} onClose={() => setSelectedDay(null)} />
      )}
    </div>
  );
}
