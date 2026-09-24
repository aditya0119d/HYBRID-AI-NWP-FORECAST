"use client";

import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

type Location = {
  id: string;
  name: string;
  lat: number;
  lon: number;
  region: string;
};

type WeatherPoint = {
  valid_time: string;
  forecast_hour: number;
  temperature: number | null;
  precipitation: number | null;
  humidity: number | null;
  pressure: number | null;
  wind_speed: number | null;
};

type ForecastData = {
  location: Location;
  generated_at: string;
  raw_gfs: WeatherPoint[];
  raw_ecmwf: WeatherPoint[];
  corrected_gfs: WeatherPoint[];
  corrected_ecmwf: WeatherPoint[];
  final_blend: WeatherPoint[];
  observed: WeatherPoint[];
};

export default function Home() {
  const [location, setLocation] = useState("");
  const [locations, setLocations] = useState<Location[]>([]);
  const [data, setData] = useState<ForecastData | null>(null);
  const [loadingLocations, setLoadingLocations] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadLocations() {
      try {
        const response = await fetch(`${API_URL}/locations`);

        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }

        const result: Location[] = await response.json();

        setLocations(result);

        if (result.length > 0) {
          setLocation(result[0].id);
        }
      } catch (err) {
        console.error(err);
        setError("Could not load forecast locations.");
      } finally {
        setLoadingLocations(false);
      }
    }

    loadLocations();
  }, []);

  async function loadForecast() {
    if (!location) return;

    setLoading(true);
    setError("");
    setData(null);

    try {
      const response = await fetch(
        `${API_URL}/forecast/${location}`
      );

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const result: ForecastData = await response.json();

      setData(result);
    } catch (err) {
      console.error(err);
      setError(
        "Could not connect to the forecasting backend."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={styles.page}>
      {/* HEADER */}
      <header style={styles.header}>
        <div>
          <div style={styles.badge}>
            AI + NWP • DISASTER FORECASTING
          </div>

          <h1 style={styles.title}>
            Hybrid AI–NWP
            <br />
            Forecast System
          </h1>

          <p style={styles.subtitle}>
            Multi-model weather forecasting using GFS + ECMWF,
            XGBoost bias correction and hybrid blending.
          </p>
        </div>

        <div style={styles.status}>
          <span style={styles.statusDot} />
          SYSTEM ONLINE
        </div>
      </header>

      {/* LOCATION CONTROL */}
      <section style={styles.controlCard}>
        <div>
          <label style={styles.label}>
            FORECAST LOCATION
          </label>

          <select
            value={location}
            onChange={(e) => {
              setLocation(e.target.value);
              setData(null);
              setError("");
            }}
            disabled={loadingLocations}
            style={styles.select}
          >
            {loadingLocations ? (
              <option>Loading locations...</option>
            ) : locations.length === 0 ? (
              <option>No locations available</option>
            ) : (
              locations.map((city) => (
                <option key={city.id} value={city.id}>
                  {city.name}
                </option>
              ))
            )}
          </select>
        </div>

        <button
          onClick={loadForecast}
          disabled={loading || !location}
          style={styles.button}
        >
          {loading ? "Generating..." : "Generate Forecast"}
        </button>
      </section>

      {/* ERROR */}
      {error && (
        <div style={styles.error}>
          ⚠ {error}
        </div>
      )}

      {/* FORECAST */}
      {data && (
        <>
          {/* LOCATION */}
          <section style={styles.locationCard}>
            <div>
              <p style={styles.smallText}>
                FORECAST LOCATION
              </p>

              <h2 style={styles.locationName}>
                {data.location.name}
              </h2>

              <p style={styles.coordinates}>
                {data.location.lat}° N
                &nbsp; • &nbsp;
                {data.location.lon}° E
              </p>
            </div>

            <div style={styles.region}>
              {data.location.region}
            </div>
          </section>

          {/* METRICS */}
          <section style={styles.grid}>
            <Metric
              title="GFS"
              value={getTemperature(data.raw_gfs)}
            />

            <Metric
              title="ECMWF"
              value={getTemperature(data.raw_ecmwf)}
            />

            <Metric
              title="GFS + ML"
              value={getTemperature(data.corrected_gfs)}
            />

            <Metric
              title="ECMWF + ML"
              value={getTemperature(data.corrected_ecmwf)}
            />

            <Metric
              title="HYBRID FORECAST"
              value={getTemperature(data.final_blend)}
              highlight
            />
          </section>

          {/* PIPELINE */}
          <section style={styles.panel}>
            <h2 style={styles.panelTitle}>
              Forecast Pipeline
            </h2>

            <div style={styles.pipeline}>
              <PipelineStep text="GFS" />
              <Arrow />
              <PipelineStep text="XGBoost Bias Correction" />
              <Arrow />
              <PipelineStep text="ECMWF" />
              <Arrow />
              <PipelineStep text="XGBoost Bias Correction" />
              <Arrow />
              <PipelineStep text="Hybrid Blending" />
            </div>
          </section>

          {/* FORECAST DETAILS */}
          <section style={styles.panel}>
            <h2 style={styles.panelTitle}>
              Forecast Details
            </h2>

            <div style={styles.detailGrid}>
              <Detail
                label="Temperature"
                value={`${getTemperature(data.final_blend)} °C`}
              />

              <Detail
                label="Precipitation"
                value={`${getValue(data.final_blend, "precipitation")} mm`}
              />

              <Detail
                label="Humidity"
                value={`${getValue(data.final_blend, "humidity")} %`}
              />

              <Detail
                label="Pressure"
                value={`${getValue(data.final_blend, "pressure")} hPa`}
              />

              <Detail
                label="Wind Speed"
                value={`${getValue(data.final_blend, "wind_speed")} m/s`}
              />

              <Detail
                label="Forecast Time"
                value={formatTime(data.final_blend)}
              />
            </div>
          </section>

          {/* DATA */}
          <section style={styles.panel}>
            <h2 style={styles.panelTitle}>
              Forecast Data
            </h2>

            <pre style={styles.json}>
              {JSON.stringify(data, null, 2)}
            </pre>
          </section>
        </>
      )}

      {!data && !loading && !error && (
        <section style={styles.empty}>
          <div style={styles.emptyIcon}>◈</div>
          <h2>Ready for Forecast</h2>
          <p>
            Select a location and generate a forecast using
            the hybrid AI–NWP pipeline.
          </p>
        </section>
      )}

      <footer style={styles.footer}>
        Hybrid AI–NWP Multi-Model Forecast Blending System
        <br />
        GFS • ECMWF • XGBoost • Hybrid Ensemble
      </footer>
    </main>
  );
}

/* =========================
   HELPERS
========================= */

function getTemperature(rows: WeatherPoint[]) {
  if (!rows || rows.length === 0) return "--";

  const value = rows[0]?.temperature;

  return value == null
    ? "--"
    : value.toFixed(2);
}

function getValue(
  rows: WeatherPoint[],
  field: keyof WeatherPoint
) {
  if (!rows || rows.length === 0) return "--";

  const value = rows[0]?.[field];

  if (typeof value !== "number") return "--";

  return value.toFixed(2);
}

function formatTime(rows: WeatherPoint[]) {
  if (!rows || rows.length === 0) return "--";

  return new Date(rows[0].valid_time).toLocaleString(
    "en-IN",
    {
      dateStyle: "medium",
      timeStyle: "short",
    }
  );
}

/* =========================
   COMPONENTS
========================= */

function Metric({
  title,
  value,
  highlight = false,
}: {
  title: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      style={{
        ...styles.metric,
        ...(highlight ? styles.highlightMetric : {}),
      }}
    >
      <p style={styles.metricTitle}>{title}</p>

      <div style={styles.metricValue}>
        {value}
      </div>

      <span style={styles.unit}>°C</span>
    </div>
  );
}

function Detail({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div style={styles.detail}>
      <span style={styles.detailLabel}>
        {label}
      </span>

      <strong style={styles.detailValue}>
        {value}
      </strong>
    </div>
  );
}

function PipelineStep({ text }: { text: string }) {
  return (
    <div style={styles.pipelineStep}>
      {text}
    </div>
  );
}

function Arrow() {
  return <div style={styles.arrow}>→</div>;
}

/* =========================
   STYLES
========================= */

const styles: any = {
  page: {
    minHeight: "100vh",
    background:
      "linear-gradient(135deg,#07111f 0%,#0b1728 50%,#101c30 100%)",
    color: "#e5edf7",
    padding: "40px",
    fontFamily: "Arial, Helvetica, sans-serif",
  },

  header: {
    maxWidth: "1200px",
    margin: "0 auto",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: "30px",
  },

  badge: {
    display: "inline-block",
    padding: "7px 12px",
    borderRadius: "20px",
    background: "#12263d",
    color: "#60a5fa",
    fontSize: "12px",
    fontWeight: "700",
    letterSpacing: "1px",
  },

  title: {
    fontSize: "48px",
    lineHeight: "1.05",
    margin: "18px 0",
  },

  subtitle: {
    maxWidth: "650px",
    color: "#94a3b8",
    fontSize: "17px",
    lineHeight: "1.6",
  },

  status: {
    padding: "10px 15px",
    borderRadius: "20px",
    background: "#10291f",
    color: "#4ade80",
    fontSize: "12px",
    fontWeight: "700",
  },

  statusDot: {
    display: "inline-block",
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#4ade80",
    marginRight: "8px",
  },

  controlCard: {
    maxWidth: "1200px",
    margin: "60px auto 30px",
    padding: "26px",
    borderRadius: "18px",
    background: "#111f31",
    border: "1px solid #243b55",
    display: "flex",
    alignItems: "end",
    gap: "20px",
  },

  label: {
    display: "block",
    color: "#7891b2",
    fontSize: "12px",
    fontWeight: "700",
    marginBottom: "10px",
    letterSpacing: "1px",
  },

  select: {
    width: "240px",
    padding: "13px",
    borderRadius: "8px",
    background: "#0b1625",
    color: "#e5edf7",
    border: "1px solid #30445e",
    fontSize: "15px",
  },

  button: {
    padding: "14px 24px",
    borderRadius: "8px",
    border: "none",
    background: "#3267e8",
    color: "white",
    fontWeight: "700",
    cursor: "pointer",
  },

  error: {
    maxWidth: "1200px",
    margin: "20px auto",
    padding: "16px",
    borderRadius: "10px",
    background: "#45191b",
    border: "1px solid #8f3033",
    color: "#ff8d8d",
  },

  locationCard: {
    maxWidth: "1200px",
    margin: "30px auto",
    padding: "25px",
    borderRadius: "18px",
    background: "#111f31",
    border: "1px solid #243b55",
    display: "flex",
    justifyContent: "space-between",
  },

  smallText: {
    color: "#7891b2",
    fontSize: "12px",
    letterSpacing: "1px",
  },

  locationName: {
    fontSize: "32px",
    margin: "8px 0",
  },

  coordinates: {
    color: "#94a3b8",
  },

  region: {
    color: "#60a5fa",
    fontWeight: "700",
  },

  grid: {
    maxWidth: "1200px",
    margin: "30px auto",
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fit,minmax(180px,1fr))",
    gap: "16px",
  },

  metric: {
    padding: "22px",
    borderRadius: "14px",
    background: "#111f31",
    border: "1px solid #243b55",
  },

  highlightMetric: {
    border: "1px solid #3267e8",
    background: "#132b4c",
  },

  metricTitle: {
    color: "#7891b2",
    fontSize: "13px",
    fontWeight: "700",
  },

  metricValue: {
    fontSize: "30px",
    fontWeight: "700",
    marginTop: "15px",
  },

  unit: {
    color: "#94a3b8",
  },

  panel: {
    maxWidth: "1200px",
    margin: "30px auto",
    padding: "25px",
    borderRadius: "18px",
    background: "#111f31",
    border: "1px solid #243b55",
  },

  panelTitle: {
    marginTop: 0,
  },

  pipeline: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexWrap: "wrap",
    gap: "12px",
  },

  pipelineStep: {
    padding: "12px 18px",
    borderRadius: "8px",
    background: "#172b43",
    border: "1px solid #304b6b",
    fontSize: "13px",
  },

  arrow: {
    color: "#60a5fa",
    fontSize: "22px",
  },

  detailGrid: {
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fit,minmax(180px,1fr))",
    gap: "14px",
  },

  detail: {
    padding: "18px",
    background: "#0b1625",
    borderRadius: "10px",
    border: "1px solid #243b55",
  },

  detailLabel: {
    display: "block",
    color: "#7891b2",
    fontSize: "12px",
    marginBottom: "8px",
  },

  detailValue: {
    fontSize: "18px",
  },

  json: {
    maxHeight: "400px",
    overflow: "auto",
    padding: "20px",
    borderRadius: "10px",
    background: "#07111f",
    color: "#9cc4ff",
    fontSize: "12px",
  },

  empty: {
    maxWidth: "1200px",
    margin: "30px auto",
    padding: "70px",
    textAlign: "center",
    borderRadius: "18px",
    background: "#111f31",
    border: "1px solid #243b55",
  },

  emptyIcon: {
    fontSize: "40px",
    color: "#3267e8",
  },

  footer: {
    maxWidth: "1200px",
    margin: "60px auto 0",
    paddingTop: "25px",
    borderTop: "1px solid #243b55",
    textAlign: "center",
    color: "#607895",
    fontSize: "13px",
    lineHeight: "1.8",
  },
};