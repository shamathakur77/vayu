"""The nightly orchestrator. One command, zero manual steps:

    python -m pipeline.run_nightly

Runs at about 03:15 IST via GitHub Actions and does, in order:

1. UPDATE OBSERVATIONS with backfill. For each city, every missing
   day in the trailing week is fetched from OpenAQ (monitor truth),
   falling back to Open-Meteo CAMS (flagged source=cams). Weather
   aggregates and FIRMS fire counts are attached the same way. A
   night that failed leaves gaps; the next night repairs them.
2. GRADE every matured, ungraded prediction and refresh the
   scoreboard. Radical accountability means this step can never be
   skipped while forecasting continues.
3. TRAIN and FORECAST per city for today (lead 1) and tomorrow
   (lead 2), with 80 percent quantile bands, and append to the
   append-only ledger.
4. BUILD data/latest.json with card payloads (current CPCB reading,
   predictions, yesterday's grade, one sourced comparison).

Any city error is collected, everything else still completes and
commits, and the process exits nonzero at the end so the Actions
run fails loudly and emails the owner. Silent skips do not exist.
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from .aqi import aqi_category, pm25_to_aqi
from .config import (CITIES, LATEST_FILE, RUNLOG_FILE)
from . import store
from .comparisons import pick_comparison
from .fetch_cpcb import current_pm25
from .fetch_firms import fire_count
from .fetch_openaq import daily_pm25
from .fetch_openmeteo import cams_pm25_daily, weather_daily
from .grade import grade_matured, update_scoreboard, yesterday_grade
from .model import (FEATURE_ORDER, _day_features, calibrate_band,
                    fit_city, persistence_with_band, predict_with_band)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vayu")

BACKFILL_DAYS = 7


def ist_now():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def update_observations(city_key, errors):
    """Fill missing observation rows for the trailing week."""
    today = ist_now().date()
    obs = store.load_obs(city_key)
    have = {r["date"]: r for _, r in obs.iterrows()} if len(obs) else {}
    try:
        cams = cams_pm25_daily(city_key, past_days=BACKFILL_DAYS)
    except Exception as e:
        errors.append(f"{city_key}: open-meteo cams failed: {e}")
        cams = {}
    try:
        weather = weather_daily(city_key, past_days=BACKFILL_DAYS,
                                forecast_days=3)
    except Exception as e:
        errors.append(f"{city_key}: open-meteo weather failed: {e}")
        weather = {}
    bbox = CITIES[city_key]["fires_bbox"]
    rows = []
    for back in range(1, BACKFILL_DAYS + 1):
        d = (today - timedelta(days=back)).strftime("%Y-%m-%d")
        existing = have.get(d)
        needs_pm = (existing is None or pd.isna(existing.get("pm25"))
                    or str(existing.get("source")) == "cams")
        needs_weather = (existing is None
                         or pd.isna(existing.get("wind_speed_mean")))
        needs_fires = bbox and (existing is None
                                or pd.isna(existing.get("fire_count")))
        if not (needs_pm or needs_weather or needs_fires):
            continue
        row = {"date": d}
        if needs_pm:
            val, n = daily_pm25(city_key, d)
            if val is not None:
                row.update(pm25=round(val, 1), source="openaq",
                           n_sensors=n)
            elif d in cams:
                row.update(pm25=cams[d], source="cams", n_sensors=0)
            else:
                errors.append(f"{city_key}: no observation for {d} "
                              "from openaq or cams")
        if needs_weather and d in weather:
            row.update(weather[d])
        if needs_fires:
            fc = fire_count(bbox, day_range=1, date=d)
            if fc is not None:
                row["fire_count"] = fc
        rows.append(row)
    if rows:
        obs = store.upsert_obs(city_key, rows)
    return obs


def forecast_city(city_key, obs_df, errors):
    """Train on the city's history and emit lead 1 and lead 2
    predictions. Returns list of prediction dicts (also appended to
    the ledger by the caller)."""
    today = ist_now().date()
    df = obs_df.dropna(subset=["pm25"]).copy()
    if df.empty:
        errors.append(f"{city_key}: no observations at all, cannot forecast")
        return []
    df["dt"] = pd.to_datetime(df["date"])
    obs_series = df.set_index("dt")["pm25"].astype(float)
    fires_series = pd.to_numeric(df.set_index("dt")["fire_count"],
                                 errors="coerce")
    try:
        weather = weather_daily(city_key, past_days=2, forecast_days=3)
    except Exception as e:
        errors.append(f"{city_key}: forecast weather failed: {e}")
        weather = {}
    bbox = CITIES[city_key]["fires_bbox"]
    recent_fires = fire_count(bbox, day_range=2) if bbox else None

    models, version = fit_city(df)
    scores = store.load_scores()
    city_scores = (scores[scores["city"] == city_key]
                   if len(scores) else scores)
    preds = []
    for lead in (1, 2):
        target = today + timedelta(days=lead - 1)
        # lead definition: made this morning, target today (lead 1)
        # and tomorrow (lead 2); the anchor observation is from
        # target - lead, i.e. yesterday for both.
        tdt = pd.Timestamp(target)
        base_lo, base_mid, base_hi = persistence_with_band(
            obs_series, tdt, lead)
        if models is not None:
            feats = _day_features(
                obs_series, fires_series,
                weather.get(target.strftime("%Y-%m-%d")), tdt, lead)
            if recent_fires is not None:
                feats["fires"] = recent_fires
            lo, mid, hi = predict_with_band(models, feats)
        else:
            lo, mid, hi = base_lo, base_mid, base_hi
        if len(city_scores):
            lead_sc = city_scores[city_scores["lead_days"] == lead]
            residuals = (lead_sc["pm25_actual"]
                         - lead_sc["pm25_pred"]).to_numpy()
            lo, hi = calibrate_band(lo, mid, hi, residuals)
        preds.append({
            "made_at_utc": store.utcnow_iso(),
            "made_on_ist": today.strftime("%Y-%m-%d"),
            "city": city_key,
            "target_date": target.strftime("%Y-%m-%d"),
            "lead_days": lead,
            "pm25_pred": round(mid, 1),
            "pm25_lo": round(lo, 1),
            "pm25_hi": round(hi, 1),
            "aqi_pred": pm25_to_aqi(mid),
            "model_version": version,
            "baseline_pred": round(base_mid, 1),
        })
    return preds


def archive_context(city_key, obs_by_city, current_value):
    """Context for archive-type comparisons."""
    ctx = {}
    obs = obs_by_city.get(city_key)
    cutoff = (ist_now().date() - timedelta(days=90)).strftime("%Y-%m-%d")
    if obs is not None and len(obs):
        season = obs[(obs["date"] >= cutoff)].dropna(subset=["pm25"])
        if len(season):
            worst = season.loc[season["pm25"].idxmax()]
            ctx["season_worst"] = {"value": float(worst["pm25"]),
                                   "date": worst["date"],
                                   "n_days": int(len(season))}
            ctx["season_n_days"] = int(len(season))
            if current_value is not None:
                ctx["season_percentile"] = round(
                    100.0 * (season["pm25"] < current_value).mean(), 1)
    cleanest = None
    for ck in CITIES:
        o = obs_by_city.get(ck)
        if o is None or not len(o):
            continue
        recent = o.dropna(subset=["pm25"]).tail(1)
        if not len(recent):
            continue
        v = float(recent.iloc[0]["pm25"])
        if cleanest is None or v < cleanest["value"]:
            cleanest = {"city_key": ck, "name": CITIES[ck]["name"],
                        "value": v}
    ctx["cleanest_city"] = cleanest
    return ctx


def build_latest(obs_by_city, preds_by_city, errors):
    today = ist_now().date()
    y_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    out = {"generated_at": store.utcnow_iso(),
           "date_ist": today.strftime("%Y-%m-%d"),
           "cities": {}}
    for city_key, cfg in CITIES.items():
        entry = {"name": cfg["name"]}
        now_val, now_upd = None, None
        try:
            now_val, now_upd = current_pm25(cfg["cpcb_city"])
        except Exception as e:
            errors.append(f"{city_key}: cpcb snapshot failed: {e}")
        if now_val is not None:
            entry["now"] = {"pm25": round(now_val, 1),
                            "source": "cpcb",
                            "last_update": now_upd}
        else:
            obs = obs_by_city.get(city_key)
            if obs is not None and len(obs):
                last = obs.dropna(subset=["pm25"]).tail(1)
                if len(last):
                    entry["now"] = {"pm25": float(last.iloc[0]["pm25"]),
                                    "source": f"latest daily "
                                              f"({last.iloc[0]['source']})",
                                    "last_update": last.iloc[0]["date"]}
        preds = preds_by_city.get(city_key, [])
        for p in preds:
            slot = "today" if p["lead_days"] == 1 else "tomorrow"
            entry[slot] = {
                "target_date": p["target_date"],
                "pm25": p["pm25_pred"],
                "lo": p["pm25_lo"],
                "hi": p["pm25_hi"],
                "aqi": p["aqi_pred"],
                "category": aqi_category(p["aqi_pred"]),
            }
        entry["yesterday"] = yesterday_grade(city_key, y_date)
        headline = None
        for p in preds:
            if p["lead_days"] == 2:
                headline = p["pm25_pred"]
        basis = (entry.get("now") or {}).get("pm25") or headline
        if basis:
            ctx = archive_context(city_key, obs_by_city, basis)
            comp = pick_comparison(basis, city_key, today, ctx)
            if comp:
                entry["comparison"] = comp
        out["cities"][city_key] = entry
    LATEST_FILE.write_text(json.dumps(out, indent=2))
    return out


def main():
    errors = []
    obs_by_city = {}
    for city_key in CITIES:
        try:
            obs_by_city[city_key] = update_observations(city_key, errors)
        except Exception as e:
            errors.append(f"{city_key}: observation update crashed: {e}")
            obs_by_city[city_key] = store.load_obs(city_key)

    try:
        n_graded = grade_matured(obs_by_city)
        log.info("graded %d matured predictions", n_graded)
    except Exception as e:
        errors.append(f"grading crashed: {e}")
    try:
        update_scoreboard()
    except Exception as e:
        errors.append(f"scoreboard crashed: {e}")

    # dedupe guard: a manual re-run on the same IST day must not
    # append duplicate ledger rows; the first committed prediction
    # for a (city, target, lead) made tonight stands.
    existing = store.load_predictions()
    today_str = ist_now().strftime("%Y-%m-%d")
    already = set()
    if len(existing):
        tonight = existing[existing["made_on_ist"] == today_str]
        already = set(zip(tonight["city"], tonight["target_date"],
                          tonight["lead_days"].astype(int)))

    preds_by_city = {}
    for city_key in CITIES:
        try:
            preds = forecast_city(city_key, obs_by_city[city_key], errors)
            fresh = [p for p in preds
                     if (p["city"], p["target_date"], p["lead_days"])
                     not in already]
            if fresh and len(fresh) < len(preds):
                log.info("%s: %d of tonight's predictions already in "
                         "the ledger, keeping the originals", city_key,
                         len(preds) - len(fresh))
            if fresh:
                store.append_predictions(fresh)
            if preds:
                preds_by_city[city_key] = preds
        except Exception as e:
            errors.append(f"{city_key}: forecast crashed: {e}")

    try:
        build_latest(obs_by_city, preds_by_city, errors)
    except Exception as e:
        errors.append(f"latest.json build crashed: {e}")

    runlog = {"finished_at": store.utcnow_iso(),
              "errors": errors,
              "cities_forecast": sorted(preds_by_city)}
    RUNLOG_FILE.write_text(json.dumps(runlog, indent=2))

    if errors:
        log.error("NIGHTLY RUN COMPLETED WITH %d ERROR(S):", len(errors))
        for e in errors:
            log.error("  - %s", e)
        sys.exit(1)
    log.info("nightly run clean")


if __name__ == "__main__":
    main()
