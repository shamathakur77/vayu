"""OpenAQ v3: graded daily truth and the CPCB fallback.

OpenAQ mirrors India's official monitor network (CPCB and state
boards) plus other providers, and unlike the data.gov.in endpoint it
serves history, which makes it the source we grade predictions
against.

Definition of the observed value for (city, day), also stated in the
README scoreboard section:

    For each PM2.5 sensor within 25 km of the city point, take the
    daily summary for that local calendar day. Keep sensors whose
    daily coverage is at least 75 percent. The observed value is the
    MEDIAN of those sensor daily means. If no sensor qualifies, the
    value falls back to the Open-Meteo CAMS model daily mean and is
    flagged source=cams so nobody mistakes a model for a monitor.

Sensor discovery results are cached in data/openaq_sensors.json so a
normal night costs a handful of API calls per city.

Key: set OPENAQ_KEY in the environment. Free signup, see README.
"""

import json
import logging
import os

from .config import CITIES, DATA_DIR, OPENAQ_BASE
from .http_util import get_with_retries

log = logging.getLogger("vayu")

SENSOR_CACHE = DATA_DIR / "openaq_sensors.json"
PM25_PARAMETER_ID = 2
MAX_SENSORS_PER_CITY = 8
MIN_COVERAGE_PCT = 75.0


def _headers():
    key = os.environ.get("OPENAQ_KEY", "").strip()
    if not key:
        log.error("OPENAQ_KEY not set. OpenAQ fallback and grading "
                  "will be skipped this run. Add the secret, next "
                  "night's backfill will repair the gap.")
        return None
    return {"X-API-Key": key}


def _load_cache():
    if SENSOR_CACHE.exists():
        return json.loads(SENSOR_CACHE.read_text())
    return {}


def _save_cache(cache):
    SENSOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SENSOR_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def discover_sensors(city_key: str, force=False):
    """Find PM2.5 sensor ids near the city. Cached across runs."""
    cache = _load_cache()
    if not force and city_key in cache and cache[city_key]:
        return cache[city_key]
    headers = _headers()
    if headers is None:
        return []
    city = CITIES[city_key]
    try:
        r = get_with_retries(
            f"{OPENAQ_BASE}/locations",
            params={
                "coordinates": f"{city['lat']},{city['lon']}",
                "radius": str(city["openaq_radius_m"]),
                "parameters_id": str(PM25_PARAMETER_ID),
                "limit": "100",
            },
            headers=headers,
        )
        results = r.json().get("results", [])
    except Exception as e:
        log.error("OpenAQ location discovery failed for %s: %s", city_key, e)
        return cache.get(city_key, [])

    sensors = []
    for loc in results:
        for s in loc.get("sensors", []):
            param = s.get("parameter", {})
            if param.get("id") == PM25_PARAMETER_ID:
                sensors.append({
                    "sensor_id": s["id"],
                    "location": loc.get("name"),
                    "provider": (loc.get("provider") or {}).get("name"),
                })
    sensors = sensors[:MAX_SENSORS_PER_CITY]
    if sensors:
        cache[city_key] = sensors
        _save_cache(cache)
        log.info("OpenAQ: %d PM2.5 sensors cached for %s",
                 len(sensors), city_key)
    else:
        log.warning("OpenAQ: no PM2.5 sensors found near %s", city_key)
    return sensors


def daily_pm25_range(city_key: str, date_from: str, date_to: str):
    """Observed daily mean PM2.5 for a date range (bootstrap and
    backfill). Returns {date: (median_value, n_sensors)}. One API
    call per sensor, so a multi-year seed is still cheap."""
    headers = _headers()
    if headers is None:
        return {}
    sensors = discover_sensors(city_key)
    if not sensors:
        return {}
    per_date = {}
    for s in sensors:
        try:
            r = get_with_retries(
                f"{OPENAQ_BASE}/sensors/{s['sensor_id']}/days",
                params={"date_from": date_from, "date_to": date_to,
                        "limit": "1000"},
                headers=headers,
                attempts=2, timeout=60,
            )
            for row in r.json().get("results", []):
                period = row.get("period", {})
                local_date = ((period.get("datetimeFrom") or {})
                              .get("local", ""))[:10]
                if not local_date:
                    continue
                cov = (row.get("coverage") or {}).get("percentComplete")
                val = (row.get("value")
                       if row.get("value") is not None
                       else (row.get("summary") or {}).get("avg"))
                if val is None:
                    continue
                if cov is not None and cov < MIN_COVERAGE_PCT:
                    continue
                per_date.setdefault(local_date, []).append(float(val))
        except Exception as e:
            log.warning("OpenAQ range fetch failed for sensor %s: %s",
                        s["sensor_id"], e)
    out = {}
    for d, values in per_date.items():
        values.sort()
        n = len(values)
        median = (values[n // 2] if n % 2 == 1
                  else 0.5 * (values[n // 2 - 1] + values[n // 2]))
        out[d] = (median, n)
    return out


def daily_pm25(city_key: str, date_iso: str):
    """Observed daily mean PM2.5 for one local calendar day.

    Returns (value, n_sensors) or (None, 0). Median across qualifying
    sensors of the sensor daily mean."""
    headers = _headers()
    if headers is None:
        return None, 0
    sensors = discover_sensors(city_key)
    if not sensors:
        return None, 0
    values = []
    for s in sensors:
        try:
            r = get_with_retries(
                f"{OPENAQ_BASE}/sensors/{s['sensor_id']}/days",
                params={"date_from": date_iso, "date_to": date_iso,
                        "limit": "3"},
                headers=headers,
                attempts=2,
            )
            for row in r.json().get("results", []):
                period = row.get("period", {})
                local_date = ((period.get("datetimeFrom") or {})
                              .get("local", ""))[:10]
                if local_date and local_date != date_iso:
                    continue
                cov = (row.get("coverage") or {}).get("percentComplete")
                val = (row.get("value")
                       if row.get("value") is not None
                       else (row.get("summary") or {}).get("avg"))
                if val is None:
                    continue
                if cov is not None and cov < MIN_COVERAGE_PCT:
                    log.info("OpenAQ sensor %s on %s below coverage "
                             "(%s%%), skipped", s["sensor_id"],
                             date_iso, cov)
                    continue
                values.append(float(val))
        except Exception as e:
            log.warning("OpenAQ sensor %s failed for %s: %s",
                        s["sensor_id"], date_iso, e)
    if not values:
        log.warning("OpenAQ: no qualifying sensor-days for %s on %s",
                    city_key, date_iso)
        return None, 0
    values.sort()
    n = len(values)
    median = (values[n // 2] if n % 2 == 1
              else 0.5 * (values[n // 2 - 1] + values[n // 2]))
    return median, n
