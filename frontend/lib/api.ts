const API_BASE_URL = "http://127.0.0.1:8000";
export interface Location {
  id: string;
  name: string;
  lat?: number;
  lon?: number;
  region?: string;
}

export interface ForecastResponse {
  [key: string]: any;
}

export interface ModelPerformance {
  [key: string]: any;
}

export async function getLocations(): Promise<Location[]> {
  const response = await fetch(`${API_BASE_URL}/locations`);

  if (!response.ok) {
    throw new Error("Failed to fetch locations");
  }

  return response.json();
}

export async function getForecast(
  locationId: string
): Promise<ForecastResponse> {
  const response = await fetch(
    `${API_BASE_URL}/forecast/${locationId}`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch forecast");
  }

  return response.json();
}

export async function getForecastWeights(
  locationId: string
): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/forecast/${locationId}/weights`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch forecast weights");
  }

  return response.json();
}

export async function getModelPerformance(): Promise<ModelPerformance> {
  const response = await fetch(
    `${API_BASE_URL}/model-performance`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch model performance");
  }

  return response.json();
}

export async function getForecastHistory(
  locationId: string
): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/forecast/${locationId}/history`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch forecast history");
  }

  return response.json();
}