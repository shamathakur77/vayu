"""NASA FIRMS active fire counts.

Feeds the Delhi model during stubble season (Punjab and Haryana box)
and, with a much smaller expected effect, Pune and Nashik (western
Maharashtra box). Free MAP_KEY, limit 5000 transactions per 10 min,
max 5 days per query. One call per distinct bbox per night, far
inside the limit.

If FIRMS_KEY is missing or the call fails, fire_count is recorded as
missing (empty), never as zero. Zero means "we looked and there were
no fires". Missing means "we could not look". The model imputes
missing as 0 but the data file keeps the difference honest.
"""

import csv
import io
import logging
import os

from .config import FIRMS_BASE
from .http_util import get_with_retries

log = logging.getLogger("vayu")

SOURCE = "VIIRS_SNPP_NRT"
SOURCE_ARCHIVE = "VIIRS_SNPP_SP"  # standard processing, for history

# Circuit breaker. GitHub Actions runners sometimes cannot reach the
# FIRMS host at all (connection errors, not throttling), and every
# doomed attempt wastes minutes. After BREAKER_LIMIT consecutive
# total failures in one process, further FIRMS calls return None
# immediately and one loud warning explains why. Fire features are a
# bonus signal, never a dependency: missing counts are imputed as
# zero by the model and recorded as missing in the data.
BREAKER_LIMIT = 2
_consecutive_failures = 0
_breaker_announced = False


def _breaker_open():
    global _breaker_announced
    if _consecutive_failures >= BREAKER_LIMIT:
        if not _breaker_announced:
            log.error("FIRMS unreachable from this runner (%d consecutive "
                      "failures). Skipping all remaining fire fetches this "
                      "run; counts will be recorded as missing.",
                      _consecutive_failures)
            _breaker_announced = True
        return True
    return False


def _note_result(ok: bool):
    global _consecutive_failures
    _consecutive_failures = 0 if ok else _consecutive_failures + 1


def fire_counts_by_day(bbox: str, day_range: int, date: str,
                       source: str = SOURCE_ARCHIVE):
    """Per-day detection counts for a date-anchored range, parsed
    from the acq_date column. Used by the bootstrap and backtest.
    Returns {date: count} or None on failure."""
    if not bbox:
        return None
    key = os.environ.get("FIRMS_KEY", "").strip()
    if not key:
        return None
    if _breaker_open():
        return None
    url = f"{FIRMS_BASE}/{key}/{source}/{bbox}/{day_range}/{date}"
    try:
        r = get_with_retries(url, attempts=1, timeout=30)
    except Exception as e:
        log.error("FIRMS history fetch failed (%s %s): %s", bbox, date, e)
        _note_result(False)
        return None
    _note_result(True)
    text = r.text.strip()
    if text.lower().startswith("invalid"):
        log.error("FIRMS rejected the request: %s", text[:200])
        return None
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return {}
    header = rows[0]
    try:
        di = header.index("acq_date")
    except ValueError:
        log.error("FIRMS csv missing acq_date column: %s", header)
        return None
    counts = {}
    for row in rows[1:]:
        if len(row) > di:
            counts[row[di]] = counts.get(row[di], 0) + 1
    return counts


def fire_count(bbox: str, day_range: int = 2, date: str = None):
    """Count VIIRS active fire detections in bbox over the trailing
    day_range days (optionally anchored at date YYYY-MM-DD).
    Returns int or None on failure."""
    if not bbox:
        return None
    key = os.environ.get("FIRMS_KEY", "").strip()
    if not key:
        log.warning("FIRMS_KEY not set, fire features recorded as missing")
        return None
    if _breaker_open():
        return None
    url = f"{FIRMS_BASE}/{key}/{SOURCE}/{bbox}/{day_range}"
    if date:
        url = f"{url}/{date}"
    try:
        r = get_with_retries(url, attempts=2, timeout=45, backoff=3)
    except Exception as e:
        log.error("FIRMS fetch failed for bbox %s: %s", bbox, e)
        _note_result(False)
        return None
    _note_result(True)
    text = r.text.strip()
    if text.lower().startswith("invalid"):
        log.error("FIRMS rejected the request: %s", text[:200])
        return None
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return 0
    # first row is the header
    count = max(0, len(rows) - 1)
    log.info("FIRMS: %d detections in %s over %dd", count, bbox, day_range)
    return count
