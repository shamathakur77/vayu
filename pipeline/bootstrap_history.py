"""One-time history seeding. Run once at setup (workflow_dispatch),
safe to re-run: it only fills what is missing.

    python -m pipeline.bootstrap_history --start 2022-07-01

Builds data/observations/{city}.csv from:
  1. OpenAQ daily monitor values where available (source=openaq)
  2. Open-Meteo CAMS reanalysis for the gaps (source=cams)
  3. ERA5 archive daily weather aggregates
  4. FIRMS archived fire counts (VIIRS standard processing) for
     cities with a fire box, fetched in 5-day chunks

Without this seed the nightly job still works, it just spends its
first months in persistence-fallback mode while history accumulates.
The backtest requires the seed.
"""

import argparse
import logging
from datetime import date, datetime, timedelta

from .config import CITIES
from . import store
from .fetch_firms import fire_counts_by_day
from .fetch_openaq import daily_pm25_range
from .fetch_openmeteo import cams_pm25_history, weather_history

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vayu")

CHUNK_DAYS = 90

# Fire history is only fetched for the burning season (Sep to Feb).
# Outside those months the fire feature is missing and the model
# imputes it as zero, which is documented and close to the truth for
# the crop residue signal this feature exists to capture. This plus
# fetching each distinct bounding box exactly once (Pune and Nashik
# share one) keeps the bootstrap comfortably inside a free runner's
# patience. Full-year counts arrive naturally as the nightly job
# appends real days.
FIRE_MONTHS = {9, 10, 11, 12, 1, 2}


def daterange_chunks(start: date, end: date, step_days: int):
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=step_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_fires_for_bbox(bbox: str, start: date, end: date):
    """Per-day fire counts for one bbox, burning season months only,
    in 5-day chunks. Returns {date_str: count}."""
    fires = {}
    for a, b in daterange_chunks(start, end, 5):
        if a.month not in FIRE_MONTHS and b.month not in FIRE_MONTHS:
            continue
        got = fire_counts_by_day(bbox, (b - a).days + 1,
                                 a.strftime("%Y-%m-%d"))
        if got is not None:
            for d in daterange_chunks(a, b, 1):
                ds = d[0].strftime("%Y-%m-%d")
                fires[ds] = got.get(ds, 0)
    return fires


def seed_city(city_key: str, start: date, end: date, fires_by_bbox=None):
    log.info("=== seeding %s from %s to %s ===", city_key, start, end)
    obs = store.load_obs(city_key)
    have_pm = {r["date"] for _, r in obs.iterrows()
               if r.get("pm25") == r.get("pm25")} if len(obs) else set()

    openaq = {}
    for a, b in daterange_chunks(start, end, 360):
        openaq.update(daily_pm25_range(
            city_key, a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    log.info("%s: openaq has %d monitor days", city_key, len(openaq))

    cams, weather = {}, {}
    for a, b in daterange_chunks(start, end, CHUNK_DAYS):
        a_s, b_s = a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")
        try:
            cams.update(cams_pm25_history(city_key, a_s, b_s))
        except Exception as e:
            log.error("%s: cams history %s..%s failed: %s",
                      city_key, a_s, b_s, e)
        try:
            weather.update(weather_history(city_key, a_s, b_s))
        except Exception as e:
            log.error("%s: weather history %s..%s failed: %s",
                      city_key, a_s, b_s, e)
    log.info("%s: cams %d days, weather %d days",
             city_key, len(cams), len(weather))

    fires = {}
    bbox = CITIES[city_key]["fires_bbox"]
    if bbox:
        fires = (fires_by_bbox or {}).get(bbox, {})
        log.info("%s: fire counts for %d days (shared bbox fetch)",
                 city_key, len(fires))

    rows = []
    cur = start
    while cur <= end:
        d = cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)
        if d in have_pm:
            continue
        row = {"date": d}
        if d in openaq:
            row.update(pm25=round(openaq[d][0], 1), source="openaq",
                       n_sensors=openaq[d][1])
        elif d in cams:
            row.update(pm25=cams[d], source="cams", n_sensors=0)
        else:
            continue
        if d in weather:
            row.update(weather[d])
        if d in fires:
            row["fire_count"] = fires[d]
        rows.append(row)
    if rows:
        store.upsert_obs(city_key, rows)
    log.info("%s: wrote %d new observation days", city_key, len(rows))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-07-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = (datetime.strptime(args.end, "%Y-%m-%d").date() if args.end
           else date.today() - timedelta(days=1))
    bboxes = {c["fires_bbox"] for c in CITIES.values() if c["fires_bbox"]}
    fires_by_bbox = {}
    for bbox in sorted(bboxes):
        log.info("fetching fire history for bbox %s", bbox)
        fires_by_bbox[bbox] = fetch_fires_for_bbox(bbox, start, end)
        log.info("bbox %s: %d fire days", bbox, len(fires_by_bbox[bbox]))
    total = 0
    for city_key in CITIES:
        total += seed_city(city_key, start, end, fires_by_bbox)
    log.info("bootstrap complete, %d rows written", total)


if __name__ == "__main__":
    main()
