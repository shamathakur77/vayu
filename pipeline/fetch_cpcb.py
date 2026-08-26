"""CPCB real time AQI snapshot via data.gov.in.

This endpoint is the official one and also the flakiest one, so the
fetch is deliberately paranoid:

* tries the API key as a query parameter AND as a header
* tries json format explicitly
* retries with backoff on every variant
* logs the exact failure for every dead variant
* returns None instead of raising, so the caller can fall back to
  OpenAQ without killing the run

The endpoint returns a CURRENT snapshot only (per station, per
pollutant). It has no history, which is why it is used for the
"right now" number on cards and as a same-day sanity check, while
OpenAQ provides the graded daily truth. See README.

Key: set DATA_GOV_IN_KEY in the environment (free signup at
data.gov.in). Without it we fall back to the public sample key,
which data.gov.in throttles hard (fine for one call per night, and
documented in the README).
"""

import logging
import os

import pandas as pd

from .config import CPCB_URLS
from .http_util import get_with_retries

log = logging.getLogger("vayu")

# Public sample key published by data.gov.in for testing. Heavily
# throttled. Registering a personal key is step one in the README.
SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"


def _api_key():
    key = os.environ.get("DATA_GOV_IN_KEY", "").strip()
    if not key:
        log.warning("DATA_GOV_IN_KEY not set, using throttled sample key")
        return SAMPLE_KEY
    return key


def fetch_city_snapshot(cpcb_city: str):
    """Return a DataFrame of the current CPCB snapshot for one city,
    or None if every variant failed. Columns: station, pollutant_id,
    avg_value, last_update."""
    key = _api_key()
    base_params = {
        "format": "json",
        "limit": "500",
        "filters[city]": cpcb_city,
    }
    variants = []
    for url in CPCB_URLS:
        variants.append((url, {**base_params, "api-key": key}, None))
        variants.append((url, base_params, {"api-key": key}))

    for url, params, headers in variants:
        try:
            r = get_with_retries(url, params=params, headers=headers,
                                 attempts=2, backoff=8)
            payload = r.json()
        except Exception as e:
            log.warning("CPCB variant failed (%s, header_auth=%s): %s",
                        url, headers is not None, e)
            continue
        records = payload.get("records") or []
        if not records:
            log.warning("CPCB variant returned zero records for %s "
                        "(status=%s)", cpcb_city, payload.get("status"))
            continue
        df = pd.DataFrame.from_records(records)
        needed = {"station", "pollutant_id", "avg_value", "last_update"}
        if not needed.issubset(df.columns):
            log.warning("CPCB response missing columns, got %s",
                        list(df.columns))
            continue
        df["avg_value"] = pd.to_numeric(df["avg_value"], errors="coerce")
        log.info("CPCB snapshot ok for %s: %d records", cpcb_city, len(df))
        return df

    log.error("CPCB: all variants failed for %s, caller should fall "
              "back to OpenAQ", cpcb_city)
    return None


def current_pm25(cpcb_city: str):
    """Median PM2.5 avg_value across the city's stations right now,
    with the freshest last_update seen. Returns (value, last_update)
    or (None, None)."""
    df = fetch_city_snapshot(cpcb_city)
    if df is None:
        return None, None
    pm = df[df["pollutant_id"].str.upper().str.contains("PM2.5", regex=False)]
    pm = pm.dropna(subset=["avg_value"])
    if pm.empty:
        log.warning("CPCB snapshot for %s has no PM2.5 rows", cpcb_city)
        return None, None
    value = float(pm["avg_value"].median())
    last_update = str(pm["last_update"].max())
    return value, last_update
