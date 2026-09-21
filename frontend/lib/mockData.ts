import type {
  ForecastHistoryResponse,
  ForecastResponse,
  ForecastWeightsResponse,
  Location,
  ModelPerformanceResponse,
  WeatherPoint,
} from "../types/api";

/* -------------------------------------------------------------------------- */
/* Deterministic mock locations                                                */
/* -------------------------------------------------------------------------- */

export const MOCK_LOCATIONS: Location[] = [
  {
    id: "mumbai",
    name: "Mumbai",
    lat: 19.076,
    lon: 72.8777,
    region: "West Coast",
  },
  {
    id: "chennai",
    name: "Chennai",
    lat: 13.0827,
    lon: 80.2707,
    region: "East Coast",
  },
  {
    id: "delhi",
    name: "Delhi",
    lat: 28.6139,
    lon: 77.209,
    region: "North India",
  },
  {
    id: "kolkata",
    name: "Kolkata",
    lat: 22.5726,
    lon: 88.3639,
    region: "East India",
  },
  {
    id: "jaipur",
    name: "Jaipur",
    lat: 26.9124,
    lon: 75.7873,
    region: "Northwest India",
  },
  {
    id: "bengaluru",
    name: "Bengaluru",
    lat: 12.9716,
    lon: 77.5946,
    region: "South India",
  },
  {
    id: "ahmedabad",
    name: "Ahmedabad",
    lat: 23.0225,
    lon: 72.5714,
    region: "West India",
  },
  {
    id: "shimla",
    name: "Shimla",
    lat: 31.1048,
    lon: 77.1734,
    region: "Himalayan",
  },
];

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

const locationBias: Record<string, number> = {
  mumbai: 2,
  chennai: 3,
  delhi: 5,
  kolkata: 4,
  jaipur: 7,
  bengaluru: 1,
  ahmedabad: 6,
  shimla: -8,
};

function round(value: number, decimals = 1): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function createWeatherSeries(locationId: string): WeatherPoint[] {
  const baseTemperature = 29 + (locationBias[locationId] ?? 0);

  return Array.from({ length: 21 }, (_, index) => {
    const forecastHour = index * 6;

    const temperature =
      baseTemperature +
      2.5 * Math.sin(index / 3) -
      forecastHour * 0.015;

    const precipitation =
      index % 5 === 0
        ? round(2 + Math.abs(Math.sin(index)) * 10)
        : index % 3 === 0
          ? round(Math.abs(Math.sin(index)) * 2)
          : 0;

    const humidity = 62 + Math.sin(index / 2) * 10;

    const pressure = 1008 + Math.cos(index / 3) * 4;

    const windSpeed = 8 + Math.abs(Math.sin(index / 2)) * 7;

    const validTime = new Date(
      Date.UTC(2026, 8, 22, index * 6)
    ).toISOString();

    return {
      valid_time: validTime,
      forecast_hour: forecastHour,
      temperature: round(temperature),
      precipitation: round(precipitation),
      humidity: round(humidity),
      pressure: round(pressure),
      wind_speed: round(windSpeed),
    };
  });
}

/* -------------------------------------------------------------------------- */
/* Forecast                                                                    */
/* -------------------------------------------------------------------------- */

export function getMockForecast(locationId: string): ForecastResponse {
  const location =
    MOCK_LOCATIONS.find((item) => item.id === locationId) ??
    MOCK_LOCATIONS[0];

  const base = createWeatherSeries(location.id);

  const rawGfs = base.map((point) => ({
    ...point,
    temperature: round(point.temperature + 1.1),
    precipitation: round(point.precipitation * 1.12),
  }));

  const rawEcmwf = base.map((point) => ({
    ...point,
    temperature: round(point.temperature + 0.3),
    precipitation: round(point.precipitation * 0.82),
  }));

  const correctedGfs = rawGfs.map((point) => ({
    ...point,
    temperature: round(point.temperature - 0.8),
    precipitation: round(point.precipitation * 0.94),
  }));

  const correctedEcmwf = rawEcmwf.map((point) => ({
    ...point,
    temperature: round(point.temperature - 0.15),
    precipitation: round(point.precipitation * 1.05),
  }));

  const finalBlend = correctedGfs.map((point, index) => ({
    ...point,
    temperature: round(
      point.temperature * 0.45 +
        correctedEcmwf[index].temperature * 0.55
    ),
    precipitation: round(
      point.precipitation * 0.45 +
        correctedEcmwf[index].precipitation * 0.55
    ),
  }));

  const observed = base.map((point, index) => ({
    ...point,
    temperature: round(point.temperature - 0.2 + Math.sin(index) * 0.15),
    precipitation: round(point.precipitation * 0.95),
  }));

  return {
    location,
    generated_at: new Date().toISOString(),
    raw_gfs: rawGfs,
    raw_ecmwf: rawEcmwf,
    corrected_gfs: correctedGfs,
    corrected_ecmwf: correctedEcmwf,
    final_blend: finalBlend,
    observed,
  };
}

/* -------------------------------------------------------------------------- */
/* Model trust                                                                 */
/* -------------------------------------------------------------------------- */

export function getMockWeights(
  locationId: string
): ForecastWeightsResponse {
  const location =
    MOCK_LOCATIONS.find((item) => item.id === locationId) ??
    MOCK_LOCATIONS[0];

  const gfsTrust =
    locationId === "mumbai"
      ? 44
      : locationId === "shimla"
        ? 61
        : locationId === "jaipur"
          ? 68
          : 53;

  const ecmwfTrust = 100 - gfsTrust;

  const trend_30day = Array.from({ length: 30 }, (_, index) => {
    const variation = Math.sin(index / 4) * 5;

    return {
      date: new Date(
        Date.UTC(2026, 7, 24 + index)
      ).toISOString().slice(0, 10),

      gfs_trust_pct: round(
        Math.max(20, Math.min(80, gfsTrust + variation))
      ),

      ecmwf_trust_pct: round(
        Math.max(20, Math.min(80, ecmwfTrust - variation))
      ),
    };
  });

  return {
    location,
    current: {
      gfs_trust_pct: gfsTrust,
      ecmwf_trust_pct: ecmwfTrust,
      basis:
        "Trust is based on recent model error patterns for this location.",
    },
    trend_30day,
  };
}

/* -------------------------------------------------------------------------- */
/* Model performance                                                           */
/* -------------------------------------------------------------------------- */

export function getMockModelPerformance(): ModelPerformanceResponse {
  return {
    by_model: [
      {
        model_name: "GFS",
        mae: 2.8,
        rmse: 4.1,
        bias: 1.2,
      },
      {
        model_name: "ECMWF",
        mae: 2.3,
        rmse: 3.6,
        bias: -0.5,
      },
      {
        model_name: "Corrected GFS",
        mae: 1.9,
        rmse: 2.8,
        bias: 0.2,
      },
      {
        model_name: "Corrected ECMWF",
        mae: 1.7,
        rmse: 2.5,
        bias: -0.1,
      },
      {
        model_name: "Final Blend",
        mae: 1.5,
        rmse: 2.2,
        bias: 0.0,
      },
    ],

    by_lead_time: [
      {
        bucket: "0–24h",
        model_name: "Final Blend",
        mae: 1.1,
        rmse: 1.7,
      },
      {
        bucket: "24–48h",
        model_name: "Final Blend",
        mae: 1.4,
        rmse: 2.1,
      },
      {
        bucket: "48–72h",
        model_name: "Final Blend",
        mae: 1.7,
        rmse: 2.5,
      },
      {
        bucket: "72–120h",
        model_name: "Final Blend",
        mae: 2.0,
        rmse: 2.9,
      },
    ],

    by_season: [
      {
        season: "Winter",
        model_name: "Final Blend",
        mae: 1.2,
        rmse: 1.8,
      },
      {
        season: "Summer",
        model_name: "Final Blend",
        mae: 1.6,
        rmse: 2.3,
      },
      {
        season: "Monsoon",
        model_name: "Final Blend",
        mae: 1.8,
        rmse: 2.7,
      },
      {
        season: "Post-monsoon",
        model_name: "Final Blend",
        mae: 1.4,
        rmse: 2.1,
      },
    ],
  };
}

/* -------------------------------------------------------------------------- */
/* Historical forecast performance                                             */
/* -------------------------------------------------------------------------- */

export function getMockHistory(
  locationId: string
): ForecastHistoryResponse {
  const base = createWeatherSeries(locationId);

  return base.slice(0, 14).map((point, index) => ({
    valid_time: point.valid_time,
    predicted: round(point.temperature + 0.8),
    observed: round(point.temperature),
    model_used: index % 2 === 0 ? "Final Blend" : "Corrected ECMWF",
  }));
}