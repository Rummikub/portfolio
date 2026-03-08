# Water Quality Prediction

Predicting water quality parameters across South Africa's rivers using satellite imagery and machine learning.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![React](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb)

## Overview

This project predicts three water quality indicators — **Total Alkalinity**, **Electrical Conductance**, and **Dissolved Reactive Phosphorus** — by fusing ground-truth measurements with Landsat satellite imagery and TerraClimate data. An interactive web dashboard visualizes the predictions and methodology.

Built for the [EY Open Science Data Challenge 2026 — Optimizing Clean Water Supply](https://challenge.ey.com/challenges/2026-optimizing-clean-water-supply/rules).

## Live Dashboard

🌐 [View the dashboard](https://wqpey2026.lovable.app)

## Project Structure

```
├── src/                  # React web dashboard (Vite + TypeScript)
│   ├── components/       # UI components
│   ├── data/             # Prediction & model data for the dashboard
│   └── pages/            # Page layouts
├── public/               # Static assets
└── README.md
```

> **Note:** Data, model artifacts, and notebooks are excluded from this repository for data privacy. Refer to the [EY Challenge page](https://challenge.ey.com/challenges/2026-optimizing-clean-water-supply/rules) for details.

## Methodology

- **Data Sources:** ~200 monitoring stations (2011–2015), Landsat spectral bands, TerraClimate PET
- **Features:** 29 engineered features including spectral indices (NDWI, NDMI, MNDWI), band ratios, climate interactions, log/squared transforms, and temporal/geographic encodings
- **Models:** 12 candidates benchmarked via 5-fold CV; best performers: XGBoost (TA R²=0.85, EC R²=0.87), ExtraTrees (DRP R²=0.72)
- **Preprocessing:** RobustScaler, winsorized targets (1st–99th percentile), smart clipping

## Targets

| Parameter | Unit |
|-----------|------|
| Total Alkalinity | mg/L |
| Electrical Conductance | mS/m |
| Dissolved Reactive Phosphorus | µg/L |

## Run Locally

```bash
npm install
npm run dev
```

## Tech Stack

- React 18 + TypeScript
- Vite
- Tailwind CSS
- Recharts & Leaflet
- shadcn/ui

## License

This project is for educational and research purposes.
