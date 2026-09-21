/**
 * Frozen frontend/backend API contract.
 *
 * These types represent the future FastAPI response shapes.
 * Mock data and real API responses must conform to these types.
 */

/* -------------------------------------------------------------------------- */
/* Shared Types                                                               */
/* -------------------------------------------------------------------------- */

export type WeatherPoint = {
  valid_time: string;
  forecast_hour: number;
  temperature: number;
  precipitation: number;
  humidity: number;
  pressure: number;
  wind_speed: number;
};

export type Location = {
  id: string;
  name: string;
  lat: number;
  lon: number;
  region: string;
};

/* -------------------------------------------------------------------------- */
/* GET /locations                                                             */
/* -------------------------------------------------------------------------- */

export type LocationsResponse = Location[];

/* -------------------------------------------------------------------------- */
/* GET /forecast/{location_id}                                                */
/* -------------------------------------------------------------------------- */

export type ForecastResponse = {
  location: Location;
  generated_at: string;

  raw_gfs: WeatherPoint[];
  raw_ecmwf: WeatherPoint[];

  corrected_gfs: WeatherPoint[];
  corrected_ecmwf: WeatherPoint[];

  final_blend: WeatherPoint[];

  observed: WeatherPoint[];
};

/* -------------------------------------------------------------------------- */
/* GET /forecast/{location_id}/weights                                        */
/* -------------------------------------------------------------------------- */

export type ModelTrust = {
  gfs_trust_pct: number;
  ecmwf_trust_pct: number;
  basis: string;
};

export type ModelTrustTrendPoint = {
  date: string;
  gfs_trust_pct: number;
  ecmwf_trust_pct: number;
};

export type ForecastWeightsResponse = {
  location: Location;
  current: ModelTrust;
  trend_30day: ModelTrustTrendPoint[];
};

/* -------------------------------------------------------------------------- */
/* GET /model-performance                                                      */
/* -------------------------------------------------------------------------- */

export type ModelPerformance = {
  model_name: string;
  mae: number;
  rmse: number;
  bias: number;
};

export type LeadTimePerformance = {
  bucket: string;
  model_name: string;
  mae: number;
  rmse: number;
};

export type SeasonalPerformance = {
  season: string;
  model_name: string;
  mae: number;
  rmse: number;
};

export type ModelPerformanceResponse = {
  by_model: ModelPerformance[];
  by_lead_time: LeadTimePerformance[];
  by_season: SeasonalPerformance[];
};

/* -------------------------------------------------------------------------- */
/* GET /forecast/{location_id}/history                                        */
/* -------------------------------------------------------------------------- */

export type ForecastHistoryPoint = {
  valid_time: string;
  predicted: number;
  observed: number;
  model_used: string;
};

export type ForecastHistoryResponse = ForecastHistoryPoint[];