"""Comparison engine: turns a PM2.5 value plus the library into one
card-ready line with its source and caveat.

Selection is deterministic (day of year + city index) so cards
rotate through the library with no randomness, which keeps nightly
runs reproducible. Entries that do not apply (wrong city, value
below a min threshold, archive too thin) are skipped for the day.
"""

import json
import logging
from datetime import date as date_cls

from .aqi import aqi_category, pm25_to_aqi
from .config import CITIES, COMPARISONS_FILE

log = logging.getLogger("vayu")

# US EPA PM2.5 24h breakpoints, 2024 update (AirNow TAD)
EPA_CATEGORIES = [
    (0.0, 9.0, "Good"),
    (9.1, 35.4, "Moderate"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups"),
    (55.5, 125.4, "Unhealthy"),
    (125.5, 225.4, "Very Unhealthy"),
    (225.5, float("inf"), "Hazardous"),
]


def epa_category(pm25):
    for lo, hi, cat in EPA_CATEGORIES:
        if lo <= pm25 <= hi:
            return cat
    return "Hazardous" if pm25 > 225.4 else "Good"


def _fmt(x):
    if x >= 10:
        return str(int(round(x)))
    if x >= 3:
        v = round(x * 2) / 2
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"{round(x, 1):g}"


def load_library():
    return json.loads(COMPARISONS_FILE.read_text())["comparisons"]


def _compute(entry, pm25, city_key, archive_ctx):
    """Returns dict of placeholders or None if entry not usable."""
    c = entry["compute"]
    op = c["op"]
    city_name = CITIES[city_key]["name"]
    if c.get("min_value") and pm25 < c["min_value"]:
        return None
    if op == "divide":
        return {"result": _fmt(pm25 / c["divisor"])}
    if op == "divide_scale":
        return {"result": _fmt(pm25 / c["divisor"] * c["scale"])}
    if op == "multiple":
        m = pm25 / c["threshold"]
        if m < 1.2:
            return None
        return {"result": _fmt(m)}
    if op == "life_years":
        yrs = c["per10"] * max(0.0, pm25 - c["baseline"]) / 10.0
        if yrs < 1.0:
            return None
        return {"result": _fmt(yrs)}
    if op == "epa_category":
        return {"result": epa_category(pm25)}
    if op == "cpcb_category":
        aqi = pm25_to_aqi(pm25)
        return {"result": aqi_category(aqi), "aqi": str(aqi)}
    if op == "static":
        return {}
    if op == "archive_worst":
        w = (archive_ctx or {}).get("season_worst")
        if not w or w.get("n_days", 0) < 14:
            return None
        relation = "close to" if pm25 >= 0.85 * w["value"] else "below"
        return {"result": _fmt(w["value"]), "result_date": w["date"],
                "relation": relation}
    if op == "archive_percentile":
        pct = (archive_ctx or {}).get("season_percentile")
        n = (archive_ctx or {}).get("season_n_days", 0)
        if pct is None or n < 14 or pct < 50:
            return None
        return {"result": str(int(round(pct)))}
    if op == "cross_city":
        cc = (archive_ctx or {}).get("cleanest_city")
        if not cc or cc["city_key"] == city_key or not cc.get("value"):
            return None
        ratio = pm25 / cc["value"]
        if ratio < 1.5:
            return None
        return {"result": _fmt(ratio), "other_city": cc["name"]}
    log.warning("unknown comparison op %s", op)
    return None


def pick_comparison(pm25, city_key, on_date: date_cls, archive_ctx=None):
    """Deterministically pick one usable comparison for the day and
    return {text, source_name, source_url, caveat, id}."""
    lib = load_library()
    usable = []
    for entry in lib:
        applies = entry.get("applies_to", ["all"])
        if "all" not in applies and city_key not in applies:
            continue
        placeholders = _compute(entry, pm25, city_key, archive_ctx)
        if placeholders is None:
            continue
        usable.append((entry, placeholders))
    if not usable:
        return None
    city_idx = list(CITIES).index(city_key)
    choice, placeholders = usable[
        (on_date.timetuple().tm_yday + city_idx) % len(usable)]
    text = choice["phrase"]
    values = {"city": CITIES[city_key]["name"], **placeholders}
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return {
        "id": choice["id"],
        "text": text,
        "source_name": choice["source_name"],
        "source_url": choice["source_url"],
        "caveat": choice["caveat"],
    }
