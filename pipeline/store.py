"""Flat file store. The repo is the database and the audit trail.

observations/{city}.csv   one row per IST day:
    date, pm25, source (openaq|cams), n_sensors, fire_count,
    plus daily weather aggregate columns.

predictions/predictions.csv   append-only ledger:
    made_at_utc, made_on_ist, city, target_date, lead_days,
    pm25_pred, pm25_lo, pm25_hi, aqi_pred, model_version, baseline_pred

scores/scores.csv   one row per graded prediction:
    city, target_date, lead_days, pm25_pred, pm25_lo, pm25_hi,
    baseline_pred, pm25_actual, obs_source, abs_error, pct_error,
    baseline_abs_error, in_band, graded_on

Rows are never rewritten, only appended (grading appends to scores;
observations may be filled in late by backfill, newest source wins
only if the old row was cams and the new one is openaq).
"""

import csv
import logging
from datetime import datetime, timezone

import pandas as pd

from .config import OBS_DIR, PRED_FILE, SCORES_FILE

log = logging.getLogger("vayu")

OBS_COLUMNS = ["date", "pm25", "source", "n_sensors", "fire_count",
               "wind_speed_mean", "wind_speed_min", "temp_mean",
               "temp_min", "rh_mean", "pressure_mean", "precip_sum",
               "blh_min", "blh_mean"]

PRED_COLUMNS = ["made_at_utc", "made_on_ist", "city", "target_date",
                "lead_days", "pm25_pred", "pm25_lo", "pm25_hi",
                "aqi_pred", "model_version", "baseline_pred"]

SCORE_COLUMNS = ["city", "target_date", "lead_days", "pm25_pred",
                 "pm25_lo", "pm25_hi", "baseline_pred", "pm25_actual",
                 "obs_source", "abs_error", "pct_error",
                 "baseline_abs_error", "in_band", "graded_on"]


def obs_path(city_key):
    return OBS_DIR / f"{city_key}.csv"


def load_obs(city_key):
    p = obs_path(city_key)
    if not p.exists():
        return pd.DataFrame(columns=OBS_COLUMNS)
    df = pd.read_csv(p, dtype={"date": str})
    return df


def upsert_obs(city_key, rows):
    """rows: list of dicts keyed by OBS_COLUMNS, one per date.
    An existing openaq row is never downgraded to cams. Missing
    fields on an existing row are filled if the new row has them."""
    df = load_obs(city_key)
    by_date = {r["date"]: dict(r) for _, r in df.iterrows()} if len(df) else {}
    for row in rows:
        d = row["date"]
        old = by_date.get(d)
        if old is None:
            by_date[d] = {c: row.get(c) for c in OBS_COLUMNS}
            continue
        old_src = str(old.get("source") or "")
        new_src = str(row.get("source") or "")
        replace_pm = (pd.isna(old.get("pm25"))
                      or (old_src == "cams" and new_src == "openaq"))
        for c in OBS_COLUMNS:
            if c == "date":
                continue
            newval = row.get(c)
            if newval is None or (isinstance(newval, float) and pd.isna(newval)):
                continue
            if c in ("pm25", "source", "n_sensors"):
                if replace_pm:
                    old[c] = newval
            elif pd.isna(old.get(c)) or old.get(c) is None:
                old[c] = newval
        by_date[d] = old
    out = pd.DataFrame([by_date[d] for d in sorted(by_date)],
                       columns=OBS_COLUMNS)
    p = obs_path(city_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out


def load_predictions():
    if not PRED_FILE.exists():
        return pd.DataFrame(columns=PRED_COLUMNS)
    return pd.read_csv(PRED_FILE, dtype={"target_date": str,
                                         "made_on_ist": str})


def append_predictions(rows):
    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    exists = PRED_FILE.exists()
    with open(PRED_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_COLUMNS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in PRED_COLUMNS})
    log.info("appended %d prediction rows", len(rows))


def load_scores():
    if not SCORES_FILE.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    return pd.read_csv(SCORES_FILE, dtype={"target_date": str})


def append_scores(rows):
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    exists = SCORES_FILE.exists()
    with open(SCORES_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in SCORE_COLUMNS})
    log.info("appended %d score rows", len(rows))


def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
