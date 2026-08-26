"""Shared HTTP helper: retries with backoff and loud logging.

Every fetcher in VAYU goes through get_with_retries so that flaky
endpoints (looking at you, data.gov.in) get consistent treatment:
several attempts, exponential backoff, and a log line for every
failure. Nothing fails silently.
"""

import logging
import time

import requests

log = logging.getLogger("vayu")

DEFAULT_TIMEOUT = 30
USER_AGENT = "VAYU-forecast/1.0 (open source air quality receipts; github)"


def get_with_retries(url, params=None, headers=None, attempts=3,
                     timeout=DEFAULT_TIMEOUT, backoff=5):
    """GET with retries. Returns the Response on success, raises on
    exhaustion. Logs every failed attempt loudly."""
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    last_err = None
    for i in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            if r.status_code == 200:
                return r
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            log.warning("attempt %d/%d failed for %s -> %s",
                        i, attempts, url, last_err)
        except requests.RequestException as e:
            last_err = repr(e)
            log.warning("attempt %d/%d failed for %s -> %s",
                        i, attempts, url, last_err)
        if i < attempts:
            time.sleep(backoff * i)
    raise RuntimeError(f"all {attempts} attempts failed for {url}: {last_err}")
