"""Nightly self-grading. The receipts.

grade_matured(): every prediction whose target day now has an
observed value and no score row yet gets scored:
    abs_error      |predicted - actual|
    pct_error      100 x abs_error / actual
    in_band        actual inside [pm25_lo, pm25_hi]
    baseline_abs_error   same for the persistence baseline

update_scoreboard(): recomputes data/scoreboard.json from scores.csv.
The published headline number per city (defined identically in the
README so nobody can claim cherry-picking):

    Season accuracy = 100 minus the mean absolute percentage error
    of all 48 hour (lead 2) predictions whose target dates fall in
    the trailing 90 days and have been graded, floored at 0.

Alongside it, always published, never hidden: MAE, the persistence
baseline MAE (the skill comparison), band coverage vs the 80 percent
target, and the count of graded predictions.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import (CITIES, SCOREBOARD_FILE, SEASON_WINDOW_DAYS)
from . import store

log = logging.getLogger("vayu")


def grade_matured(obs_by_city):
    """obs_by_city: {city_key: DataFrame(date, pm25, source)}.
    Appends score rows for every matured, ungraded prediction.
    Returns the number graded."""
    preds = store.load_predictions()
    if preds.empty:
        return 0
    scores = store.load_scores()
    done = set(zip(scores["city"], scores["target_date"],
                   scores["lead_days"])) if len(scores) else set()
    today_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    new_rows = []
    for _, p in preds.iterrows():
        key = (p["city"], p["target_date"], p["lead_days"])
        if key in done:
            continue
        obs = obs_by_city.get(p["city"])
        if obs is None or obs.empty:
            continue
        row = obs[obs["date"] == p["target_date"]]
        if row.empty or pd.isna(row.iloc[0]["pm25"]):
            # not matured yet, or observation still missing
            if p["target_date"] < (today_ist - timedelta(days=7)).strftime("%Y-%m-%d"):
                log.warning("prediction for %s %s still ungradable after "
                            "7 days (no observation)", p["city"],
                            p["target_date"])
            continue
        actual = float(row.iloc[0]["pm25"])
        pred = float(p["pm25_pred"])
        base = float(p["baseline_pred"])
        abs_err = abs(pred - actual)
        new_rows.append({
            "city": p["city"],
            "target_date": p["target_date"],
            "lead_days": int(p["lead_days"]),
            "pm25_pred": round(pred, 1),
            "pm25_lo": round(float(p["pm25_lo"]), 1),
            "pm25_hi": round(float(p["pm25_hi"]), 1),
            "baseline_pred": round(base, 1),
            "pm25_actual": round(actual, 1),
            "obs_source": row.iloc[0]["source"],
            "abs_error": round(abs_err, 1),
            "pct_error": round(100.0 * abs_err / actual, 1) if actual > 0 else "",
            "baseline_abs_error": round(abs(base - actual), 1),
            "in_band": bool(float(p["pm25_lo"]) <= actual <= float(p["pm25_hi"])),
            "graded_on": store.utcnow_iso(),
        })
        done.add(key)
    if new_rows:
        store.append_scores(new_rows)
    return len(new_rows)


def _city_stats(sc: pd.DataFrame):
    lead2 = sc[sc["lead_days"] == 2]
    lead1 = sc[sc["lead_days"] == 1]

    def block(df):
        if df.empty:
            return None
        pct = pd.to_numeric(df["pct_error"], errors="coerce").dropna()
        mape = float(pct.mean()) if len(pct) else None
        return {
            "n": int(len(df)),
            "mae": round(float(df["abs_error"].mean()), 1),
            "mape": round(mape, 1) if mape is not None else None,
            "accuracy": round(max(0.0, 100.0 - mape), 1) if mape is not None else None,
            "baseline_mae": round(float(df["baseline_abs_error"].mean()), 1),
            "band_coverage": round(100.0 * float(df["in_band"].mean()), 1),
        }
    return {"lead2": block(lead2), "lead1": block(lead1)}


def update_scoreboard():
    scores = store.load_scores()
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=SEASON_WINDOW_DAYS)).strftime("%Y-%m-%d")
    board = {
        "generated_at": store.utcnow_iso(),
        "season_window_days": SEASON_WINDOW_DAYS,
        "definition": ("Season accuracy = 100 - MAPE of graded 48h "
                       "(lead 2) predictions with target dates in the "
                       "trailing 90 days. Band target: 80 percent "
                       "coverage. Full definition in the README."),
        "cities": {},
    }
    for city_key in CITIES:
        sc = scores[scores["city"] == city_key] if len(scores) else scores
        season = sc[sc["target_date"] >= cutoff] if len(sc) else sc
        board["cities"][city_key] = {
            "name": CITIES[city_key]["name"],
            "season": _city_stats(season) if len(season) else {"lead2": None, "lead1": None},
            "all_time": _city_stats(sc) if len(sc) else {"lead2": None, "lead1": None},
        }
    SCOREBOARD_FILE.write_text(json.dumps(board, indent=2))
    log.info("scoreboard updated")
    return board


def yesterday_grade(city_key, y_date):
    """The single most recent grade for the card: the lead 1
    prediction for yesterday, falling back to lead 2."""
    scores = store.load_scores()
    if scores.empty:
        return None
    sc = scores[(scores["city"] == city_key)
                & (scores["target_date"] == y_date)]
    if sc.empty:
        return None
    sc = sc.sort_values("lead_days")
    r = sc.iloc[0]
    return {
        "target_date": r["target_date"],
        "lead_days": int(r["lead_days"]),
        "predicted": float(r["pm25_pred"]),
        "actual": float(r["pm25_actual"]),
        "pct_error": float(r["pct_error"]) if r["pct_error"] != "" else None,
        "in_band": bool(r["in_band"]),
        "obs_source": r["obs_source"],
    }
