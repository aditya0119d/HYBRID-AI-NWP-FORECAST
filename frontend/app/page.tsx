"use client";

import { useEffect, useState } from "react";
import {
  getLocations,
  getForecast,
  getForecastWeights,
  getModelPerformance,
  type Location,
} from "../lib/api";

export default function Home() {
  const [locations, setLocations] = useState<Location[]>([]);
  const [selectedLocation, setSelectedLocation] = useState("");

  const [forecast, setForecast] = useState<any>(null);
  const [weights, setWeights] = useState<any>(null);
  const [performance, setPerformance] = useState<any>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // --------------------------------------------------
  // Load available locations when page opens
  // --------------------------------------------------

  useEffect(() => {
    async function loadLocations() {
      try {
        const data = await getLocations();

        setLocations(data);

        if (data.length > 0) {
          setSelectedLocation(data[0].id);
        }
      } catch (err) {
        console.error(err);
        setError(
          "Could not connect to FastAPI. Make sure the backend is running on port 8000."
        );
      }
    }

    loadLocations();
  }, []);

  // --------------------------------------------------
  // Generate forecast
  // --------------------------------------------------

  async function handleGenerateForecast() {
    if (!selectedLocation) return;

    setLoading(true);
    setError("");

    try {
      const [forecastData, weightData, performanceData] =
        await Promise.all([
          getForecast(selectedLocation),
          getForecastWeights(selectedLocation),
          getModelPerformance(),
        ]);

      setForecast(forecastData);
      setWeights(weightData);
      setPerformance(performanceData);
    } catch (err) {
      console.error(err);

      setError(
        "Failed to generate forecast. Check that FastAPI is running and the selected location exists."
      );
    } finally {
      setLoading(false);
    }
  }

  // --------------------------------------------------
  // Latest forecast point
  // --------------------------------------------------

  const latest =
    forecast?.final_blend?.length > 0
      ? forecast.final_blend[0]
      : null;

  const latestGFS =
    forecast?.raw_gfs?.length > 0
      ? forecast.raw_gfs[0]
      : null;

  const latestECMWF =
    forecast?.raw_ecmwf?.length > 0
      ? forecast.raw_ecmwf[0]
      : null;

  const latestCorrectedGFS =
    forecast?.corrected_gfs?.length > 0
      ? forecast.corrected_gfs[0]
      : null;

  const latestCorrectedECMWF =
    forecast?.corrected_ecmwf?.length > 0
      ? forecast.corrected_ecmwf[0]
      : null;

  return (
    <main
      style={{
        minHeight: "100vh",
        background: "#07111f",
        color: "#e8f0f8",
        padding: "40px",
      }}
    >
      <div
        style={{
          maxWidth: "1150px",
          margin: "0 auto",
        }}
      >
        {/* ================================================= */}
        {/* HEADER */}
        {/* ================================================= */}

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: "45px",
          }}
        >
          <div>
            <div
              style={{
                display: "inline-block",
                padding: "7px 14px",
                borderRadius: "20px",
                background: "#10283e",
                color: "#42b7f5",
                fontSize: "13px",
                fontWeight: 700,
                letterSpacing: "0.5px",
                marginBottom: "16px",
              }}
            >
              AI + NWP • DISASTER FORECASTING
            </div>

            <h1
              style={{
                fontSize: "48px",
                lineHeight: "1.05",
                margin: 0,
                maxWidth: "700px",
              }}
            >
              Hybrid AI–NWP
              <br />
              Forecast System
            </h1>

            <p
              style={{
                color: "#8da6bd",
                fontSize: "17px",
                maxWidth: "720px",
                marginTop: "20px",
              }}
            >
              Multi-model weather forecasting using GFS + ECMWF,
              XGBoost bias correction and hybrid blending.
            </p>
          </div>

          <div
            style={{
              border: "1px solid #29445c",
              borderRadius: "10px",
              padding: "10px 18px",
              color: "#6ee7a0",
              fontWeight: 700,
              fontSize: "14px",
            }}
          >
            ● SYSTEM ONLINE
          </div>
        </div>

        {/* ================================================= */}
        {/* LOCATION SELECTOR */}
        {/* ================================================= */}

        <section
          style={{
            background: "#0d1c2b",
            border: "1px solid #28445d",
            borderRadius: "14px",
            padding: "22px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "end",
            marginBottom: "30px",
          }}
        >
          <div>
            <label
              style={{
                display: "block",
                color: "#8da6bd",
                fontSize: "13px",
                fontWeight: 700,
                marginBottom: "10px",
              }}
            >
              FORECAST LOCATION
            </label>

            <select
              value={selectedLocation}
              onChange={(e) =>
                setSelectedLocation(e.target.value)
              }
              style={{
                width: "300px",
                padding: "13px",
                borderRadius: "8px",
                border: "1px solid #31516b",
                background: "#07111f",
                color: "white",
                fontSize: "15px",
              }}
            >
              {locations.map((location) => (
                <option
                  key={location.id}
                  value={location.id}
                >
                  {location.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleGenerateForecast}
            disabled={loading || !selectedLocation}
            style={{
              padding: "13px 24px",
              borderRadius: "8px",
              border: "none",
              background: "#2fa4dc",
              color: "white",
              fontWeight: 700,
              cursor: loading ? "wait" : "pointer",
            }}
          >
            {loading ? "Generating..." : "Generate Forecast"}
          </button>
        </section>

        {/* ================================================= */}
        {/* ERROR */}
        {/* ================================================= */}

        {error && (
          <div
            style={{
              background: "#35161b",
              border: "1px solid #74313b",
              color: "#ff9da8",
              padding: "16px",
              borderRadius: "10px",
              marginBottom: "25px",
            }}
          >
            {error}
          </div>
        )}

        {/* ================================================= */}
        {/* FORECAST RESULTS */}
        {/* ================================================= */}

        {forecast && latest ? (
          <>
            {/* LOCATION */}

            <section
              style={{
                background: "#0d1c2b",
                border: "1px solid #28445d",
                borderRadius: "14px",
                padding: "24px",
                marginBottom: "22px",
              }}
            >
              <div
                style={{
                  color: "#8da6bd",
                  fontSize: "13px",
                  fontWeight: 700,
                }}
              >
                FORECAST LOCATION
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginTop: "10px",
                }}
              >
                <div>
                  <h2
                    style={{
                      fontSize: "30px",
                      margin: 0,
                    }}
                  >
                    {forecast.location.name}
                  </h2>

                  <p style={{ color: "#8da6bd" }}>
                    {forecast.location.lat.toFixed(4)}° N
                    {" • "}
                    {forecast.location.lon.toFixed(4)}° E
                  </p>
                </div>

                <div
                  style={{
                    background: "#102f48",
                    color: "#58c5ff",
                    padding: "10px 18px",
                    borderRadius: "20px",
                  }}
                >
                  {forecast.location.region}
                </div>
              </div>
            </section>

            {/* MODEL CARDS */}

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(5, 1fr)",
                gap: "14px",
                marginBottom: "22px",
              }}
            >
              <ModelCard
                title="GFS"
                value={latestGFS?.temperature}
              />

              <ModelCard
                title="ECMWF"
                value={latestECMWF?.temperature}
              />

              <ModelCard
                title="GFS + ML"
                value={latestCorrectedGFS?.temperature}
              />

              <ModelCard
                title="ECMWF + ML"
                value={latestCorrectedECMWF?.temperature}
              />

              <ModelCard
                title="HYBRID FORECAST"
                value={latest.temperature}
                highlight
              />
            </div>

            {/* PIPELINE */}

            <section
              style={{
                background: "#0d1c2b",
                border: "1px solid #28445d",
                borderRadius: "14px",
                padding: "24px",
                marginBottom: "22px",
              }}
            >
              <h2>Forecast Pipeline</h2>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "12px",
                  flexWrap: "wrap",
                  marginTop: "20px",
                }}
              >
                {[
                  "GFS",
                  "XGBoost Correction",
                  "ECMWF",
                  "XGBoost Correction",
                  "Hybrid Blending",
                ].map((step, index) => (
                  <div
                    key={step + index}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "12px",
                    }}
                  >
                    <div
                      style={{
                        background: "#142a3d",
                        border: "1px solid #31516b",
                        padding: "11px 15px",
                        borderRadius: "7px",
                        fontSize: "13px",
                      }}
                    >
                      {step}
                    </div>

                    {index < 4 && (
                      <span
                        style={{
                          color: "#42b7f5",
                          fontSize: "20px",
                        }}
                      >
                        →
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* DETAILS */}

            <section
              style={{
                background: "#0d1c2b",
                border: "1px solid #28445d",
                borderRadius: "14px",
                padding: "24px",
                marginBottom: "22px",
              }}
            >
              <h2>Forecast Details</h2>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "repeat(5, 1fr)",
                  gap: "12px",
                  marginTop: "18px",
                }}
              >
                <DetailCard
                  title="Temperature"
                  value={`${latest.temperature?.toFixed(
                    2
                  )} °C`}
                />

                <DetailCard
                  title="Precipitation"
                  value={`${latest.precipitation?.toFixed(
                    2
                  )} mm`}
                />

                <DetailCard
                  title="Humidity"
                  value={`${latest.humidity?.toFixed(
                    2
                  )} %`}
                />

                <DetailCard
                  title="Pressure"
                  value={`${latest.pressure?.toFixed(
                    2
                  )} hPa`}
                />

                <DetailCard
                  title="Wind Speed"
                  value={`${latest.wind_speed?.toFixed(
                    2
                  )} m/s`}
                />
              </div>

              <div
                style={{
                  marginTop: "14px",
                  color: "#8da6bd",
                }}
              >
                Forecast time:{" "}
                <strong style={{ color: "white" }}>
                  {new Date(
                    latest.valid_time
                  ).toLocaleString()}
                </strong>
              </div>
            </section>

            {/* WEIGHTS */}

            {weights?.current && (
              <section
                style={{
                  background: "#0d1c2b",
                  border: "1px solid #28445d",
                  borderRadius: "14px",
                  padding: "24px",
                  marginBottom: "22px",
                }}
              >
                <h2>Model Trust Weights</h2>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "1fr 1fr",
                    gap: "15px",
                    marginTop: "18px",
                  }}
                >
                  <DetailCard
                    title="GFS Trust"
                    value={`${weights.current.gfs_trust_pct}%`}
                  />

                  <DetailCard
                    title="ECMWF Trust"
                    value={`${weights.current.ecmwf_trust_pct}%`}
                  />
                </div>
              </section>
            )}

            {/* PERFORMANCE */}

            {performance?.by_model && (
              <section
                style={{
                  background: "#0d1c2b",
                  border: "1px solid #28445d",
                  borderRadius: "14px",
                  padding: "24px",
                  marginBottom: "22px",
                }}
              >
                <h2>Model Performance</h2>

                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      marginTop: "18px",
                      borderCollapse: "collapse",
                    }}
                  >
                    <thead>
                      <tr>
                        <th style={th}>Model</th>
                        <th style={th}>MAE</th>
                        <th style={th}>RMSE</th>
                        <th style={th}>Bias</th>
                      </tr>
                    </thead>

                    <tbody>
                      {performance.by_model.map(
                        (model: any) => (
                          <tr key={model.model_name}>
                            <td style={td}>
                              {model.model_name}
                            </td>

                            <td style={td}>
                              {model.mae}
                            </td>

                            <td style={td}>
                              {model.rmse}
                            </td>

                            <td style={td}>
                              {model.bias}
                            </td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {/* RAW API DATA */}

            <details
              style={{
                background: "#0d1c2b",
                border: "1px solid #28445d",
                borderRadius: "14px",
                padding: "20px",
              }}
            >
              <summary
                style={{
                  cursor: "pointer",
                  fontWeight: 700,
                }}
              >
                Forecast Data — API Response
              </summary>

              <pre
                style={{
                  marginTop: "15px",
                  background: "#030a12",
                  padding: "20px",
                  borderRadius: "8px",
                  overflow: "auto",
                  maxHeight: "500px",
                  fontSize: "12px",
                }}
              >
                {JSON.stringify(
                  forecast,
                  null,
                  2
                )}
              </pre>
            </details>
          </>
        ) : (
          /* ================================================= */
          /* EMPTY STATE */
          /* ================================================= */

          <section
            style={{
              minHeight: "250px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              background: "#0d1c2b",
              border: "1px solid #28445d",
              borderRadius: "14px",
            }}
          >
            <div
              style={{
                fontSize: "42px",
                color: "#2fa4dc",
              }}
            >
              ◇
            </div>

            <h2>Ready for Forecast</h2>

            <p style={{ color: "#8da6bd" }}>
              Select a location and generate a forecast
              using the hybrid AI–NWP pipeline.
            </p>
          </section>
        )}

        {/* ================================================= */}
        {/* FOOTER */}
        {/* ================================================= */}

        <footer
          style={{
            marginTop: "45px",
            paddingTop: "25px",
            borderTop: "1px solid #233b50",
            textAlign: "center",
            color: "#617c95",
            fontSize: "13px",
          }}
        >
          Hybrid AI–NWP Multi-Model Forecast Blending System
          <br />
          GFS • ECMWF • XGBoost • Hybrid Ensemble
        </footer>
      </div>
    </main>
  );
}


// =========================================================
// SMALL COMPONENTS
// =========================================================

function ModelCard({
  title,
  value,
  highlight = false,
}: {
  title: string;
  value?: number;
  highlight?: boolean;
}) {
  return (
    <div
      style={{
        background: "#0d1c2b",
        border: highlight
          ? "1px solid #2fa4dc"
          : "1px solid #28445d",
        borderRadius: "14px",
        padding: "20px",
      }}
    >
      <div
        style={{
          color: "#8da6bd",
          fontSize: "13px",
          fontWeight: 700,
        }}
      >
        {title}
      </div>

      <div
        style={{
          fontSize: "30px",
          fontWeight: 700,
          marginTop: "10px",
        }}
      >
        {value !== undefined
          ? value.toFixed(2)
          : "--"}
      </div>

      <div style={{ color: "#718ba1" }}>
        °C
      </div>
    </div>
  );
}


function DetailCard({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <div
      style={{
        background: "#091725",
        borderRadius: "8px",
        padding: "16px",
      }}
    >
      <div
        style={{
          color: "#718ba1",
          fontSize: "13px",
        }}
      >
        {title}
      </div>

      <div
        style={{
          marginTop: "7px",
          fontWeight: 700,
          fontSize: "17px",
        }}
      >
        {value}
      </div>
    </div>
  );
}


const th: React.CSSProperties = {
  textAlign: "left",
  padding: "12px",
  borderBottom: "1px solid #28445d",
  color: "#8da6bd",
};

const td: React.CSSProperties = {
  padding: "12px",
  borderBottom: "1px solid #182d40",
};