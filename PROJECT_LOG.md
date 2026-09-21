# Hybrid AI–NWP Forecast Blending System — Project Log

## 2026-09-22 — Frontend Workstream Started

### What was built / decided

- Repository scaffold (Step 1) is complete.
- Live GFS/ECMWF ingestion is temporarily replaced by a deterministic synthetic
  data generator because of SIH prototype time constraints.
- Synthetic generator is located at:
  `src/hybrid_forecast/ingestion/synthetic_generator.py`
- Synthetic observations and NWP forecasts are generated as Parquet files.
- Synthetic data includes structured, learnable forecast biases rather than
  pure random noise.
- GFS and ECMWF use different structured bias patterns to support future
  Stage 1 correction and Stage 2 blending.
- NumPy, Pandas, and PyArrow environment issues were resolved.
- The synthetic generator has been successfully executed.
- Real GFS/ECMWF ingestion remains a future replacement and should not require
  changes to downstream ML/backend/frontend architecture.

### New frontend workstream

The frontend is intentionally being developed ahead of the backend using
mock data.

The frontend will live inside:

`frontend/`

The mock data must exactly follow the future FastAPI API contract.

Components must never directly access mock data. All data access must go
through:

`frontend/lib/api.ts`

A future environment flag will switch:

`mock data → real FastAPI API`

without rewriting components.

### Current assumptions

- Frontend stack:
  Next.js + React + TypeScript + Tailwind CSS + Recharts +
  Leaflet/OpenStreetMap.
- Light professional dashboard design.
- One primary weather/disaster-management accent color.
- Red/orange reserved for actual risk indicators.
- The dashboard is a decision-support interface, not an official warning system.
- Mock data will be deterministic and realistic-looking.
- The API contract will be frozen before UI components are developed.

### Needs verification

- Exact FastAPI implementation and response serialization will need to match
  the frozen TypeScript API contract when the backend is implemented.
- Exact production GFS/ECMWF ingestion sources remain deferred.
- Whether all requested forecast variables are available from the eventual
  real ingestion pipeline still needs verification.

### Current next step

STEP A — Freeze the TypeScript API contract.

No UI component should be built until the API types are reviewed and confirmed.