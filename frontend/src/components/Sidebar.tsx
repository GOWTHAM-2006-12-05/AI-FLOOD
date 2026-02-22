"use client";

import { ReactNode } from "react";
import clsx from "clsx";

interface SidebarProps {
  children: ReactNode;
  className?: string;
}

export function Sidebar({ children, className }: SidebarProps) {
  return (
    <aside
      className={clsx(
        "bg-surface-1 border-r border-surface-3",
        "flex flex-col gap-4 p-4 overflow-y-auto",
        className
      )}
    >
      {/* ── Logo / Title ── */}
      <div className="flex items-center gap-3 pb-3 border-b border-surface-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center">
          <svg
            className="w-5 h-5 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"
            />
          </svg>
        </div>
        <div>
          <h1 className="text-sm font-bold text-white leading-tight">
            AI Disaster Prediction
          </h1>
          <p className="text-[10px] text-gray-500 font-mono">
            Early Warning System v2.0
          </p>
        </div>
      </div>

      {children}
    </aside>
  );
}
