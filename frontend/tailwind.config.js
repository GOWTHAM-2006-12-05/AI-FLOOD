/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Risk-level colour system
        risk: {
          safe:    { DEFAULT: '#10B981', light: '#D1FAE5', dark: '#064E3B' },
          watch:   { DEFAULT: '#F59E0B', light: '#FEF3C7', dark: '#78350F' },
          warning: { DEFAULT: '#F97316', light: '#FFEDD5', dark: '#7C2D12' },
          severe:  { DEFAULT: '#EF4444', light: '#FEE2E2', dark: '#7F1D1D' },
        },
        // Dark theme surface hierarchy
        surface: {
          0: '#0A0A0F',    // deepest background
          1: '#111118',    // card background
          2: '#1A1A24',    // elevated cards
          3: '#242430',    // hover states
          4: '#2E2E3A',    // active states
        },
        accent: {
          blue:   '#3B82F6',
          cyan:   '#06B6D4',
          purple: '#8B5CF6',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-risk': 'pulse-risk 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up':   'slide-up 0.3s ease-out',
        'slide-down': 'slide-down 0.3s ease-out',
        'fade-in':    'fade-in 0.5s ease-out',
        'glow':       'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-risk': {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0.6' },
        },
        'slide-up': {
          '0%':   { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'slide-down': {
          '0%':   { transform: 'translateY(-100%)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'glow': {
          '0%':   { boxShadow: '0 0 5px rgba(59,130,246,0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(59,130,246,0.8)' },
        },
      },
    },
  },
  plugins: [],
};
