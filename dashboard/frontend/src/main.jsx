import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { RefreshCw, Wind, Gauge, Thermometer, Activity, Database, Trophy, BarChart3 } from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";
const MODELS = [
  { key: "random_forest", label: "Random Forest" },
  { key: "ridge", label: "Ridge" },
  { key: "xgboost", label: "XGBoost" },
];
const HORIZONS = [24, 48, 72];

async function api(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function formatDate(value) {
  if (!value) return "N/A";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function classForTone(tone) {
  return `tone-${tone || "muted"}`;
}

function MetricCard({ icon: Icon, label, value, suffix }) {
  return (
    <div className="metric-card glass">
      <div className="metric-icon"><Icon size={18} /></div>
      <div>
        <p>{label}</p>
        <h3>{value ?? "N/A"}{value !== null && value !== undefined && suffix ? <span>{suffix}</span> : null}</h3>
      </div>
    </div>
  );
}

function ModelSelector({ selectedModel, setSelectedModel, selectedHorizon, setSelectedHorizon }) {
  return (
    <div className="selector glass">
      <div>
        <p className="eyebrow">forecast settings</p>
        <h2>Model toggle</h2>
      </div>

      <div className="toggle-group">
        {MODELS.map((model) => (
          <button
            key={model.key}
            className={selectedModel === model.key ? "active" : ""}
            onClick={() => setSelectedModel(model.key)}
          >
            {model.label}
          </button>
        ))}
      </div>

      <div className="toggle-group small">
        {HORIZONS.map((h) => (
          <button
            key={h}
            className={selectedHorizon === h ? "active" : ""}
            onClick={() => setSelectedHorizon(h)}
          >
            {h}h
          </button>
        ))}
      </div>
    </div>
  );
}

function MetricsTable({ metrics }) {
  const rows = metrics?.items || [];
  return (
    <div className="glass section">
      <div className="section-head">
        <div>
          <p className="eyebrow">always visible</p>
          <h2>Model metric comparison</h2>
        </div>
        <BarChart3 size={20} />
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Horizon</th>
              <th>Model</th>
              <th>Best</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>R²</th>
              <th>MAPE</th>
              <th>Coverage</th>
              <th>Skill</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.horizon}-${row.model_name}`}>
                <td>{row.horizon}h</td>
                <td>{row.model_label}</td>
                <td>{row.is_best ? <span className="badge best">winner</span> : <span className="badge">compare</span>}</td>
                <td>{row.test_mae ?? "N/A"}</td>
                <td>{row.test_rmse ?? "N/A"}</td>
                <td>{row.test_r2 ?? "N/A"}</td>
                <td>{row.test_mape ? `${row.test_mape}%` : "N/A"}</td>
                <td>{row.coverage ? `${row.coverage}%` : "N/A"}</td>
                <td>{row.skill ?? "N/A"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetricsChart({ metrics }) {
  const data = (metrics?.items || []).map((row) => ({
    name: `${row.model_label} ${row.horizon}h`,
    r2: row.test_r2,
    mae: row.test_mae,
  }));

  return (
    <div className="glass section chart-section">
      <div className="section-head">
        <div>
          <p className="eyebrow">performance</p>
          <h2>R² and MAE overview</h2>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 10 }} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="r2" name="R²" radius={[8, 8, 0, 0]} />
          <Bar dataKey="mae" name="MAE" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function AQIRing({ current }) {
  const aqi = current?.aqi ?? 0;
  const percentage = Math.min(Math.max((aqi / 300) * 100, 0), 100);
  const tone = current?.category?.tone || "muted";

  return (
    <div className={`aqi-ring ${classForTone(tone)}`} style={{ "--value": `${percentage}%` }}>
      <div className="ring-inner">
        <h1>{aqi || "N/A"}</h1>
        <p>current AQI</p>
      </div>
    </div>
  );
}

function Dashboard() {
  const [selectedModel, setSelectedModel] = useState("xgboost");
  const [selectedHorizon, setSelectedHorizon] = useState(24);
  const [current, setCurrent] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);

  async function loadAll() {
    setLoading(true);
    try {
      const [currentData, metricsData, historyData, summaryData] = await Promise.all([
        api("/current"),
        api("/metrics"),
        api("/history?limit=72"),
        api("/summary"),
      ]);
      setCurrent(currentData);
      setMetrics(metricsData);
      setHistory(historyData.items || []);
      setSummary(summaryData);
    } finally {
      setLoading(false);
    }
  }

  async function runPredict() {
    const data = await api("/predict", {
      method: "POST",
      body: JSON.stringify({ model_name: selectedModel, horizon: selectedHorizon }),
    });
    setPrediction(data);
  }

  useEffect(() => {
    loadAll();
  }, []);

  useEffect(() => {
    runPredict().catch(() => setPrediction(null));
  }, [selectedModel, selectedHorizon]);

  const selectedMetric = useMemo(() => {
    return metrics?.items?.find((item) => item.model_name === selectedModel && item.horizon === selectedHorizon);
  }, [metrics, selectedModel, selectedHorizon]);

  return (
    <main>
      <header className="topbar">
        <div className="brand glass">
          <div className="logo">A</div>
          <div>
            <h3>AirLyst</h3>
            <p>Predicting Air Quality AQI of Karachi</p>
          </div>
        </div>

        <button className="refresh" onClick={loadAll}>
          <RefreshCw size={16} className={loading ? "spin" : ""} />
          Refresh
        </button>
      </header>

      <section className="hero glass">
        <div>
          <h1>Karachi, Pakistan</h1>
          <p>Live air quality monitoring, MongoDB-powered model predictions and cross-model insights</p>
        </div>
        <div className="hero-meta">
          <span>Last updated</span>
          <strong>{formatDate(current?.datetime)}</strong>
        </div>
      </section>

      <section className="grid-main">
        <div className="aqi-panel glass">
          <div className="section-head">
            <div>
              <p className="eyebrow">current air quality</p>
              <h2>{current?.category?.label || "Loading"}</h2>
              <p className="hint">{current?.category?.advice}</p>
            </div>
            <span className="live-dot">Live</span>
          </div>

          <div className="aqi-content">
            <AQIRing current={current} />

            <div className="pollutants">
              <MetricCard icon={Activity} label="PM 2.5" value={current?.pollutants?.pm25} suffix=" µg/m³" />
              <MetricCard icon={Activity} label="PM 10" value={current?.pollutants?.pm10} suffix=" µg/m³" />
              <MetricCard icon={Gauge} label="NO₂" value={current?.pollutants?.no2} suffix=" ppb" />
              <MetricCard icon={Gauge} label="SO₂" value={current?.pollutants?.so2} suffix=" ppb" />
              <MetricCard icon={Gauge} label="CO" value={current?.pollutants?.co} suffix=" ppm" />
            </div>
          </div>
        </div>

        <aside className="side-stack">
          <div className="weather-card glass">
            <p className="eyebrow">current weather</p>
            <h1>{current?.weather?.temperature ?? "N/A"}°</h1>
            <div className="weather-grid">
              <MetricCard icon={Gauge} label="Pressure" value={current?.weather?.pressure} suffix=" mb" />
              <MetricCard icon={Wind} label="Wind Speed" value={current?.weather?.wind_speed} suffix=" km/h" />
              <MetricCard icon={Thermometer} label="Humidity" value={current?.weather?.humidity} suffix="%" />
            </div>
          </div>

          <ModelSelector
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            selectedHorizon={selectedHorizon}
            setSelectedHorizon={setSelectedHorizon}
          />
        </aside>
      </section>

      <section className="forecast-grid">
        <div className="glass section prediction-card">
          <div className="section-head">
            <div>
              <p className="eyebrow">selected forecast</p>
              <h2>{prediction?.model_label || "Model"} · {selectedHorizon}h</h2>
            </div>
            <Trophy size={20} />
          </div>

          <div className={`prediction-number ${classForTone(prediction?.category?.tone)}`}>
            {prediction?.prediction ?? "N/A"}
            <span>AQI</span>
          </div>
          <p className="hint">{prediction?.category?.label || "Waiting for prediction"}</p>
          <div className="interval">
            <span>Lower: {prediction?.pi_lower ?? "N/A"}</span>
            <span>Upper: {prediction?.pi_upper ?? "N/A"}</span>
          </div>

          <div className="selected-metric">
            <div><span>MAE</span><strong>{selectedMetric?.test_mae ?? "N/A"}</strong></div>
            <div><span>R²</span><strong>{selectedMetric?.test_r2 ?? "N/A"}</strong></div>
            <div><span>Coverage</span><strong>{selectedMetric?.coverage ? `${selectedMetric.coverage}%` : "N/A"}</strong></div>
          </div>
        </div>

        <div className="glass section chart-section">
          <div className="section-head">
            <div>
              <p className="eyebrow">last 72 records</p>
              <h2>AQI history</h2>
            </div>
            <Database size={20} />
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="datetime" tickFormatter={(v) => new Date(v).getHours() + ":00"} tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip labelFormatter={formatDate} />
              <Line type="monotone" dataKey="aqi" strokeWidth={3} dot={false} name="AQI" />
              <Line type="monotone" dataKey="pm25" strokeWidth={2} dot={false} name="PM2.5" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <MetricsChart metrics={metrics} />
      <MetricsTable metrics={metrics} />

      <section className="glass section">
        <div className="section-head">
          <div>
            <p className="eyebrow">pipeline monitor</p>
            <h2>Latest backend status</h2>
          </div>
        </div>
        <div className="status-grid">
          <div><span>Feature documents</span><strong>{summary?.feature_count ?? "N/A"}</strong></div>
          {(summary?.pipeline_status || []).slice(0, 4).map((item, index) => (
            <div key={index}>
              <span>{item.pipeline} {item.step || ""}</span>
              <strong>{item.status}</strong>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<Dashboard />);
