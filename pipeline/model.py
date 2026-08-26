"""Forecast models.

Baseline (the floor everything must beat):
    persistence: prediction for target day T at lead k is the
    observed value at day T-k.

Candidate:
    gradient boosting (sklearn HistGradientBoostingRegressor) with
    quantile loss, one model set per city per lead. Quantiles 0.1,
    0.5, 0.9 give the point forecast (median) and an 80 percent
    band. Small data, trains in seconds on a free Actions runner.

Features for target day T at lead k use ONLY information available
on the morning the forecast is made (obs through T-k, weather
forecast for T, fire counts through T-k). That rule is enforced by
construction here and is what makes the backtest honest.

If a city has under MIN_TRAIN_ROWS usable rows, we fall back to
persistence with an error-quantile band and say so in the ledger
(model_version column).
"""

import logging
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import MODEL_VERSION

log = logging.getLogger("vayu")

MIN_TRAIN_ROWS = 120
QUANTILES = (0.1, 0.5, 0.9)

WEATHER_FEATS = ["wind_speed_mean", "wind_speed_min", "temp_mean",
                 "temp_min", "rh_mean", "pressure_mean", "precip_sum",
                 "blh_min"]


def _day_features(obs: pd.Series, fires: pd.Series, weather_row: dict,
                  target_date: pd.Timestamp, lead: int):
    """Feature vector for one (target_date, lead). obs and fires are
    Series indexed by Timestamp (daily). weather_row is the weather
    aggregate dict for the TARGET day (forecast or historical)."""
    anchor = target_date - pd.Timedelta(days=lead)

    def ob(days_back):
        return obs.get(anchor - pd.Timedelta(days=days_back), np.nan)

    last = ob(0)
    window7 = obs.reindex(pd.date_range(anchor - pd.Timedelta(days=6),
                                        anchor))
    feats = {
        "lag_last": last,
        "lag_1": ob(1),
        "lag_2": ob(2),
        "lag_6": ob(6),
        "roll3": obs.reindex(pd.date_range(anchor - pd.Timedelta(days=2),
                                           anchor)).mean(),
        "roll7": window7.mean(),
        "trend3": (last - ob(2)) if not (math.isnan(last) or
                                         math.isnan(ob(2))) else np.nan,
        "doy_sin": math.sin(2 * math.pi * target_date.dayofyear / 365.25),
        "doy_cos": math.cos(2 * math.pi * target_date.dayofyear / 365.25),
        "lead": lead,
        "fires": fires.get(anchor, np.nan) if fires is not None else np.nan,
    }
    wr = weather_row or {}
    for k in WEATHER_FEATS:
        v = wr.get(k)
        feats[f"w_{k}"] = np.nan if v is None else v
    return feats


FEATURE_ORDER = (["lag_last", "lag_1", "lag_2", "lag_6", "roll3",
                  "roll7", "trend3", "doy_sin", "doy_cos", "lead",
                  "fires"] + [f"w_{k}" for k in WEATHER_FEATS])


def build_training_frame(obs_df: pd.DataFrame, leads=(1, 2)):
    """Build (X, y, dates, leads) from an observations dataframe with
    date, pm25, fire_count and weather columns. Weather used for a
    training target day is that day's HISTORICAL weather, standing in
    for the forecast the model sees live (standard practice; noted
    as a limitation in the backtest report)."""
    df = obs_df.dropna(subset=["pm25"]).copy()
    df["dt"] = pd.to_datetime(df["date"])
    obs = df.set_index("dt")["pm25"].astype(float)
    fires = df.set_index("dt")["fire_count"]
    fires = pd.to_numeric(fires, errors="coerce")
    weather_by_day = {row["dt"]: {k: row.get(k) for k in WEATHER_FEATS}
                      for _, row in df.iterrows()}
    rows, ys, dts, lds = [], [], [], []
    for dt in obs.index:
        for lead in leads:
            anchor = dt - pd.Timedelta(days=lead)
            if anchor not in obs.index:
                continue
            f = _day_features(obs, fires, weather_by_day.get(dt), dt, lead)
            rows.append([f[k] for k in FEATURE_ORDER])
            ys.append(obs[dt])
            dts.append(dt)
            lds.append(lead)
    X = np.array(rows, dtype=float)
    y = np.array(ys, dtype=float)
    return X, y, pd.DatetimeIndex(dts), np.array(lds)


def train_quantile_models(X, y, seed=7):
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q,
            max_iter=250, max_depth=4, learning_rate=0.06,
            min_samples_leaf=15, l2_regularization=1.0,
            random_state=seed)
        m.fit(X, y)
        models[q] = m
    return models


def predict_with_band(models, feats: dict):
    x = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=float)
    lo = float(models[0.1].predict(x)[0])
    mid = float(models[0.5].predict(x)[0])
    hi = float(models[0.9].predict(x)[0])
    lo, mid, hi = sorted([lo, mid, hi])
    return max(1.0, lo), max(1.0, mid), max(1.0, hi)


def persistence_with_band(obs: pd.Series, target_date: pd.Timestamp,
                          lead: int):
    """Fallback: persistence point forecast with a band from the
    historical distribution of persistence errors at this lead."""
    anchor = target_date - pd.Timedelta(days=lead)
    base = obs.get(anchor, np.nan)
    if math.isnan(base):
        base = obs.dropna().iloc[-1] if len(obs.dropna()) else np.nan
    errs = []
    idx = obs.dropna().index
    for dt in idx:
        prev = dt - pd.Timedelta(days=lead)
        if prev in obs.index and not math.isnan(obs[prev]):
            errs.append(obs[dt] - obs[prev])
    if len(errs) >= 20:
        lo_off, hi_off = np.quantile(errs, [0.1, 0.9])
    else:
        lo_off, hi_off = -0.3 * base, 0.3 * base
    return (max(1.0, base + lo_off), max(1.0, base),
            max(1.0, base + hi_off))


def calibrate_band(lo, mid, hi, residuals):
    """Empirical band guard. Quantile models can undercover on small
    samples, so once at least 20 graded residuals (actual minus
    predicted) exist for this city and lead, the band is widened to
    at least the 10th to 90th percentile of recent residuals around
    the point forecast. Widening only, never narrowing: the published
    80 percent band should err on the side of humility."""
    if residuals is None or len(residuals) < 20:
        return lo, hi
    r10, r90 = np.quantile(residuals[-60:], [0.1, 0.9])
    lo = min(lo, max(1.0, mid + r10))
    hi = max(hi, mid + r90)
    return lo, hi


def fit_city(obs_df: pd.DataFrame):
    """Train city models. Returns (models_or_None, version_string)."""
    X, y, _, _ = build_training_frame(obs_df)
    if len(y) < MIN_TRAIN_ROWS:
        log.warning("only %d training rows, falling back to persistence",
                    len(y))
        return None, "persistence-fallback"
    models = train_quantile_models(X, y)
    return models, MODEL_VERSION
