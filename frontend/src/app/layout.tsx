import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Disaster Prediction — Early Warning Dashboard",
  description:
    "Hyper-local multi-disaster early warning system with real-time flood, earthquake, and cyclone monitoring.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          crossOrigin=""
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-surface-0 font-sans">{children}</body>
    </html>
  );
}
