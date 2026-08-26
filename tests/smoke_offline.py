"""Offline end-to-end smoke test. No network: every fetcher is
monkeypatched with synthetic but realistic data. Verifies the whole
chain: observations -> training -> forecast -> ledger -> grading ->
scoreboard -> latest.json -> comparison selection.

    python tests/smoke_offline.py

Run from a scratch copy of the repo, it writes into data/ and out/.
"""

import json
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from pipeline import store  # noqa: E402
from pipeline.aqi import pm25_to_aqi, aqi_category  # noqa: E402
from pipeline.config import CITIES  # noqa: E402


def synth_pm25(city_key, d: date, rng):
    """Winter-peaking series with noise, city-scaled."""
    scale = {"delhi": 170, "mumbai": 70, "pune": 75,
             "bengaluru": 45, "nashik": 60}[city_key]
    doy = d.timetuple().tm_yday
    season = 1.0 + 0.9 * math.cos(2 * math.pi * (doy - 5) / 365.25)
    ar = getattr(synth_pm25, "_prev", {}).get(city_key, scale)
    val = 0.6 * ar + 0.4 * scale * season + rng.gauss(0, scale * 0.12)
    val = max(8.0, val)
    synth_pm25._prev = getattr(synth_pm25, "_prev", {})
    synth_pm25._prev[city_key] = val
    return round(val, 1)


def seed_observations(days=540):
    rng = random.Random(7)
    end = date.today() - timedelta(days=1)
    for city_key in CITIES:
        rows = []
        for back in range(days, 0, -1):
            d = end - timedelta(days=back - 1)
            doy = d.timetuple().tm_yday
            pm = synth_pm25(city_key, d, rng)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "pm25": pm,
                "source": "openaq" if back % 9 else "cams",
                "n_sensors": 5,
                "fire_count": (int(max(0, rng.gauss(120, 80)))
                               if CITIES[city_key]["fires_bbox"]
                               and 270 < doy < 335 else 0),
                "wind_speed_mean": round(max(0.5, rng.gauss(8, 3)), 2),
                "wind_speed_min": round(max(0.1, rng.gauss(3, 1.5)), 2),
                "temp_mean": round(rng.gauss(25, 6), 2),
                "temp_min": round(rng.gauss(18, 6), 2),
                "rh_mean": round(min(100, max(10, rng.gauss(55, 15))), 2),
                "pressure_mean": round(rng.gauss(1005, 5), 2),
                "precip_sum": round(max(0, rng.gauss(0, 2)), 2),
                "blh_min": round(max(50, rng.gauss(300, 150)), 2),
                "blh_mean": round(max(100, rng.gauss(800, 300)), 2),
            })
        store.upsert_obs(city_key, rows)
    print("seeded observations:", days, "days x", len(CITIES), "cities")


def patch_fetchers():
    """Replace every network call with synthetic equivalents."""
    from pipeline import run_nightly as rn
    rng = random.Random(11)

    def fake_daily_pm25(city_key, date_iso):
        obs = store.load_obs(city_key)
        row = obs[obs["date"] == date_iso]
        if len(row) and not pd.isna(row.iloc[0]["pm25"]):
            return float(row.iloc[0]["pm25"]), 5
        return round(60 + rng.random() * 60, 1), 4

    def fake_cams(city_key, past_days=7, forecast_days=3):
        out = {}
        today = date.today()
        for back in range(1, past_days + 1):
            d = (today - timedelta(days=back)).strftime("%Y-%m-%d")
            out[d] = round(50 + rng.random() * 80, 1)
        return out

    def fake_weather(city_key, past_days=7, forecast_days=3):
        out = {}
        today = date.today()
        for off in range(-past_days, forecast_days):
            d = (today + timedelta(days=off)).strftime("%Y-%m-%d")
            out[d] = {"wind_speed_mean": 8.0, "wind_speed_min": 2.5,
                      "temp_mean": 24.0, "temp_min": 17.0,
                      "rh_mean": 60.0, "pressure_mean": 1004.0,
                      "precip_sum": 0.0, "blh_min": 250.0,
                      "blh_mean": 700.0}
        return out

    rn.daily_pm25 = fake_daily_pm25
    rn.cams_pm25_daily = fake_cams
    rn.weather_daily = fake_weather
    rn.fire_count = lambda bbox, day_range=2, date=None: 42 if bbox else None
    rn.current_pm25 = lambda cpcb_city: (round(80 + rng.random() * 200, 1),
                                         "25-08-2026 03:00:00")
    return rn


def seed_old_predictions():
    """Plant predictions for yesterday so grading has work to do."""
    y = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    for city_key in CITIES:
        obs = store.load_obs(city_key)
        row = obs[obs["date"] == y]
        actual = float(row.iloc[0]["pm25"]) if len(row) else 100.0
        for lead in (1, 2):
            pred = actual * (1 + (0.08 if lead == 1 else 0.15))
            rows.append({
                "made_at_utc": "2026-08-24T21:45:00Z",
                "made_on_ist": y, "city": city_key, "target_date": y,
                "lead_days": lead, "pm25_pred": round(pred, 1),
                "pm25_lo": round(pred * 0.75, 1),
                "pm25_hi": round(pred * 1.35, 1),
                "aqi_pred": pm25_to_aqi(pred),
                "model_version": "gbq-1.0",
                "baseline_pred": round(actual * 1.2, 1),
            })
    store.append_predictions(rows)
    print("planted", len(rows), "matured predictions")


def checks():
    from pipeline.config import (LATEST_FILE, SCOREBOARD_FILE,
                                 SCORES_FILE, PRED_FILE)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    # AQI math against CPCB breakpoints
    check(pm25_to_aqi(30) == 50, "aqi: 30 ug/m3 -> 50")
    check(pm25_to_aqi(60) == 100, "aqi: 60 ug/m3 -> 100")
    check(pm25_to_aqi(90) == 200, "aqi: 90 ug/m3 -> 200")
    check(pm25_to_aqi(120) == 300, "aqi: 120 ug/m3 -> 300")
    check(pm25_to_aqi(250) == 400, "aqi: 250 ug/m3 -> 400")
    check(pm25_to_aqi(500) == 500, "aqi: 500 ug/m3 -> 500")
    check(pm25_to_aqi(75) == 150, "aqi: 75 ug/m3 -> 150 (midband)")
    check(aqi_category(pm25_to_aqi(400)) == "Severe", "aqi: 400 is Severe")

    scores = store.load_scores()
    check(len(scores) == len(CITIES) * 2,
          f"grading: {len(scores)} rows (expected {len(CITIES) * 2})")
    if len(scores):
        r = scores.iloc[0]
        manual = abs(r["pm25_pred"] - r["pm25_actual"])
        check(abs(r["abs_error"] - manual) < 0.11,
              "grading: abs_error matches manual recompute")
        manual_pct = 100 * manual / r["pm25_actual"]
        check(abs(float(r["pct_error"]) - manual_pct) < 0.11,
              "grading: pct_error matches manual recompute")

    preds = store.load_predictions()
    fresh = preds[preds["made_on_ist"] == date.today().strftime("%Y-%m-%d")]
    check(len(fresh) == len(CITIES) * 2,
          f"forecast: {len(fresh)} fresh predictions (expected {len(CITIES) * 2})")
    gb = fresh[fresh["model_version"] == "gbq-1.0"]
    check(len(gb) == len(fresh),
          "forecast: all cities used the trained model, no fallback")
    band_ok = all(fresh["pm25_lo"] <= fresh["pm25_pred"]) and \
        all(fresh["pm25_pred"] <= fresh["pm25_hi"])
    check(band_ok, "forecast: lo <= mid <= hi everywhere")

    latest = json.loads(LATEST_FILE.read_text())
    check(len(latest["cities"]) == len(CITIES), "latest.json: all cities")
    comps = [c.get("comparison") for c in latest["cities"].values()]
    check(all(comps), "latest.json: every city got a comparison")
    check(all(c["source_url"].startswith("http") for c in comps),
          "comparisons: every one carries a source URL")
    ids = {c["id"] for c in comps}
    print("     comparison ids tonight:", sorted(ids))
    sb = json.loads(SCOREBOARD_FILE.read_text())
    d = sb["cities"]["delhi"]["season"]["lead1"]
    check(d and d["n"] >= 1, "scoreboard: delhi has graded stats")

    return ok


def main():
    seed_observations()
    seed_old_predictions()
    rn = patch_fetchers()
    rn.main.__wrapped__ = None
    try:
        rn.main()
    except SystemExit as e:
        if e.code not in (0, None):
            print("NOTE: nightly exited", e.code,
                  "(expected 0 for a clean mocked run)")
            raise
    ok = checks()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
