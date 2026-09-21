import type {
  ForecastHistoryResponse,
  ForecastResponse,
  ForecastWeightsResponse,
  LocationsResponse,
  ModelPerformanceResponse,
} from "../types/api";

import {
  getMockForecast,
  getMockWeights,
  getMockModelPerformance,
  getMockHistory,
  MOCK_LOCATIONS,
} from "./mockData";

/*
 * Frontend data-access layer.
 *
 * Components must ONLY communicate with the backend through these functions.
 *
 * Currently:
 *     NEXT_PUBLIC_USE_MOCK=true → mock data
 *
 * Later:
 *     NEXT_PUBLIC_USE_MOCK=false → FastAPI
 */

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== "false";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/* -------------------------------------------------------------------------- */
/* GET /locations                                                             */
/* -------------------------------------------------------------------------- */

export async function getLocations(): Promise<LocationsResponse> {
  if (USE_MOCK) {
    return MOCK_LOCATIONS;
  }

  const response = await fetch(`${API_BASE_URL}/locations`);

  if (!response.ok) {
    throw new Error("Failed to fetch locations");
  }

  return response.json();
}

/* -------------------------------------------------------------------------- */
/* GET /forecast/{location_id}                                                */
/* -------------------------------------------------------------------------- */

export async function getForecast(
  locationId: string
): Promise<ForecastResponse> {
  if (USE_MOCK) {
    return getMockForecast(locationId);
  }

  const response = await fetch(
    `${API_BASE_URL}/forecast/${locationId}`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch forecast");
  }

  return response.json();
}

/* -------------------------------------------------------------------------- */
/* GET /forecast/{location_id}/weights                                        */
/* -------------------------------------------------------------------------- */

export async function getForecastWeights(
  locationId: string
): Promise<ForecastWeightsResponse> {
  if (USE_MOCK) {
    return getMockWeights(locationId);
  }

  const response = await fetch(
    `${API_BASE_URL}/forecast/${locationId}/weights`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch forecast weights");
  }

  return response.json();
}

/* -------------------------------------------------------------------------- */
/* GET /model-performance                                                      */
/* -------------------------------------------------------------------------- */

export async function getModelPerformance(): Promise<ModelPerformanceResponse> {
  if (USE_MOCK) {
    return getMockModelPerformance();
  }

  const response = await fetch(`${API_BASE_URL}/model-performance`);

  if (!response.ok) {
    throw new Error("Failed to fetch model performance");
  }

  return response.json();
}

/* -------------------------------------------------------------------------- */
/* GET /forecast/{location_id}/history                                        */
/* -------------------------------------------------------------------------- */

export async function getForecastHistory(
  locationId: string
): Promise<ForecastHistoryResponse> {
  if (USE_MOCK) {
    return getMockHistory(locationId);
  }

  const response = await fetch(
    `${API_BASE_URL}/forecast/${locationId}/history`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch forecast history");
  }

  return response.json();
}