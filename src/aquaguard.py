"""Core logic for the AquaSense AI dashboard (no Streamlit dependency).

Wraps the frozen day-22 models, converts raw sensor units to the standardised
scale the models were trained on, and runs a deterministic WHO/EPA reference
audit that is independent of the ML models.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "day22_final"
EXPERIMENTS = ROOT / "experiments"

SENSORS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]


@dataclass(frozen=True)
class Sensor:
    key: str
    label: str
    unit: str
    lo: float          # slider minimum
    hi: float          # slider maximum
    step: float
    ref_lo: float | None   # reference band (None = no lower bound)
    ref_hi: float | None
    ref_text: str
    source: str


# Reference thresholds are guideline values used for an educational audit.
# They are NOT a regulatory determination - verify against local standards.
SENSOR_META: list[Sensor] = [
    Sensor("ph", "pH Acidity", "", 0.0, 14.0, 0.01, 6.5, 8.5, "6.5-8.5", "WHO operational range"),
    Sensor("Hardness", "Hardness", "mg/L", 40.0, 330.0, 1.0, None, 200.0, "\u2264 200", "Taste / scaling reference"),
    Sensor("Solids", "Total Dissolved Solids", "mg/L", 300.0, 62000.0, 50.0, None, 1000.0, "\u2264 1000", "WHO taste acceptability"),
    Sensor("Chloramines", "Chloramines", "mg/L", 0.0, 14.0, 0.05, None, 4.0, "\u2264 4", "EPA MRDL"),
    Sensor("Sulfate", "Sulfate", "mg/L", 100.0, 490.0, 1.0, None, 250.0, "\u2264 250", "EPA secondary / taste"),
    Sensor("Conductivity", "Conductivity", "\u00b5S/cm", 170.0, 760.0, 1.0, None, 400.0, "\u2264 400", "Common reference value"),
    Sensor("Organic_carbon", "Organic Carbon (TOC)", "mg/L", 2.0, 29.0, 0.1, None, 4.0, "\u2264 4", "Source-water reference"),
    Sensor("Trihalomethanes", "Trihalomethanes", "\u00b5g/L", 0.0, 130.0, 0.5, None, 80.0, "\u2264 80", "EPA MCL"),
    Sensor("Turbidity", "Turbidity", "NTU", 1.0, 7.0, 0.05, None, 5.0, "\u2264 5", "WHO guideline"),
]
META = {s.key: s for s in SENSOR_META}

# Presets are plain physical-unit readings; "Dataset Median" is added at runtime.
PRESETS: dict[str, dict[str, float]] = {
    "Mountain Spring": dict(ph=7.30, Hardness=120, Solids=350, Chloramines=2.0, Sulfate=120,
                            Conductivity=250, Organic_carbon=2.6, Trihalomethanes=25, Turbidity=1.5),
    "Municipal Tap": dict(ph=7.40, Hardness=180, Solids=800, Chloramines=3.0, Sulfate=220,
                          Conductivity=380, Organic_carbon=3.5, Trihalomethanes=60, Turbidity=1.2),
    "Industrial Runoff": dict(ph=6.20, Hardness=310, Solids=45000, Chloramines=11.0, Sulfate=440,
                              Conductivity=700, Organic_carbon=24, Trihalomethanes=110, Turbidity=6.2),
    "Acid Drainage": dict(ph=3.10, Hardness=250, Solids=38000, Chloramines=6.0, Sulfate=470,
                          Conductivity=680, Organic_carbon=12, Trihalomethanes=70, Turbidity=4.8),
    "Sensor Failure": dict(ph=0.0, Hardness=330, Solids=62000, Chloramines=0.0, Sulfate=490,
                           Conductivity=760, Organic_carbon=29, Trihalomethanes=130, Turbidity=7.0),
}


# --------------------------------------------------------------------------- loading
class Pipeline:
    """Frozen Isolation Forest + Random Forest with the recovered scaler."""

    def __init__(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.iso = joblib.load(ARTIFACTS / "final_isolation_forest.joblib")
            self.rf = joblib.load(ARTIFACTS / "final_random_forest.joblib")
        sc = json.loads((ARTIFACTS / "scaler_recovered.json").read_text())
        self.mean = np.array([sc[c]["mean"] for c in SENSORS])
        self.std = np.array([sc[c]["std"] for c in SENSORS])
        self.median_reading = self._dataset_median()

    # ---- scaling
    def to_z(self, raw: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (np.asarray(raw, dtype=float) - self.mean) / self.std

    def ph_from_z(self, z):
        return np.asarray(z) * self.std[0] + self.mean[0]

    def _dataset_median(self) -> dict[str, float]:
        raw = pd.read_csv(ROOT / "data" / "processed" / "water_quality_timestamped.csv")
        return {c: float(raw[c].median()) for c in SENSORS}

    # ---- single reading (simulator)
    def predict_one(self, reading: dict[str, float], when: datetime,
                    prev_ph: tuple[float, float]) -> dict:
        t0 = time.perf_counter()
        z = self.to_z([[reading[c] for c in SENSORS]])
        zdf = pd.DataFrame(z, columns=SENSORS)

        Xa = self.iso["imputer"].transform(zdf[self.iso["features"]])
        flag = int(self.iso["model"].predict(Xa)[0] == -1)
        score = float(self.iso["model"].decision_function(Xa)[0])

        ph_z = float(z[0, 0])
        prev_z = (np.array(prev_ph) - self.mean[0]) / self.std[0]      # [t-2, t-1]
        window = np.array([prev_z[0], prev_z[1], ph_z])
        feats = zdf.copy()
        feats["hour"] = when.hour
        feats["day_of_week"] = when.weekday()
        feats["day"] = when.day
        feats["month"] = when.month
        feats["ph_lag_1"] = prev_z[1]
        feats["ph_rolling_mean_3"] = window.mean()
        feats["ph_rolling_std_3"] = window.std(ddof=1)
        Xf = self.rf["imputer"].transform(feats[self.rf["features"]])
        next_z = float(self.rf["model"].predict(Xf)[0])
        latency_ms = (time.perf_counter() - t0) * 1000

        return dict(
            anomaly=flag, score=score, z=dict(zip(SENSORS, z[0].tolist())),
            next_ph=float(self.ph_from_z(next_z)), latency_ms=latency_ms,
        )

    # ---- batch
    def run_batch(self, df: pd.DataFrame, standardised: bool = False) -> pd.DataFrame:
        d = df.copy()
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
            d = d.sort_values("timestamp").reset_index(drop=True)
        else:
            d.insert(0, "timestamp", pd.date_range(datetime.now().replace(minute=0, second=0, microsecond=0),
                                                   periods=len(d), freq="h"))
        z = d[SENSORS].astype(float).copy() if standardised else pd.DataFrame(
            self.to_z(d[SENSORS].astype(float)), columns=SENSORS)

        Xa = self.iso["imputer"].transform(z[self.iso["features"]])
        d["anomaly_flag"] = (self.iso["model"].predict(Xa) == -1).astype(int)
        d["anomaly_score"] = self.iso["model"].decision_function(Xa)

        f = z.copy()
        f["hour"] = d["timestamp"].dt.hour.values
        f["day_of_week"] = d["timestamp"].dt.dayofweek.values
        f["day"] = d["timestamp"].dt.day.values
        f["month"] = d["timestamp"].dt.month.values
        f["ph_lag_1"] = f["ph"].shift(1)
        f["ph_rolling_mean_3"] = f["ph"].rolling(3).mean()
        f["ph_rolling_std_3"] = f["ph"].rolling(3).std()
        Xf = self.rf["imputer"].transform(f[self.rf["features"]])
        pred_z = self.rf["model"].predict(Xf)
        d["predicted_next_ph"] = self.ph_from_z(pred_z)
        d.loc[f["ph_lag_1"].isna() | f["ph_rolling_std_3"].isna(), "predicted_next_ph"] = np.nan
        raw_vals = pd.DataFrame(z.values * self.std + self.mean, columns=SENSORS) if standardised \
            else d[SENSORS].astype(float)
        d["who_violations"] = audit_frame(raw_vals).sum(axis=1).values
        return d


# --------------------------------------------------------------------------- WHO audit
def audit_reading(reading: dict[str, float]) -> list[dict]:
    rows = []
    for s in SENSOR_META:
        v = reading[s.key]
        ok = (s.ref_lo is None or v >= s.ref_lo) and (s.ref_hi is None or v <= s.ref_hi)
        rows.append(dict(key=s.key, label=s.label, unit=s.unit, value=v,
                         ref=s.ref_text, source=s.source, ok=bool(ok)))
    return rows


def audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean violation matrix (True = outside reference band)."""
    out = pd.DataFrame(index=df.index)
    for s in SENSOR_META:
        v = df[s.key].astype(float)
        bad = pd.Series(False, index=df.index)
        if s.ref_lo is not None:
            bad |= v < s.ref_lo
        if s.ref_hi is not None:
            bad |= v > s.ref_hi
        out[s.key] = bad
    return out


# --------------------------------------------------------------------------- saved results
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def baseline_check(pipe: Pipeline) -> dict:
    """Re-computes forecast metrics on the held-out test split, adding a
    train-mean baseline that the original validation did not include."""
    tr = pd.read_csv(ROOT / "data" / "processed" / "train_v1_engineered.csv")
    te = pd.read_csv(ROOT / "data" / "processed" / "test_v1_engineered.csv")

    def prep(df):
        d = df.copy()
        d["ph_next"] = d["ph"].shift(-1)
        return d.dropna(subset=["ph_next", "ph_lag_1", "ph_rolling_mean_3", "ph_rolling_std_3"]).reset_index(drop=True)

    tr_p, te_p = prep(tr), prep(te)
    y = te_p["ph_next"].values
    X = pipe.rf["imputer"].transform(te_p[pipe.rf["features"]])
    preds = {
        "Tuned Random Forest": pipe.rf["model"].predict(X),
        "Persistence (last value)": te_p["ph"].values,
        "Train-mean constant": np.full_like(y, tr_p["ph_next"].mean()),
    }
    rows = [dict(Method=k, MAE=float(np.mean(np.abs(y - p))), RMSE=float(np.sqrt(np.mean((y - p) ** 2))))
            for k, p in preds.items()]
    return dict(table=pd.DataFrame(rows), n=len(y),
                lag1_corr=float(np.corrcoef(te_p["ph"], y)[0, 1]))
