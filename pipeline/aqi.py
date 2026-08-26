"""Indian National AQI math for PM2.5.

Breakpoints are the CPCB National AQI sub-index breakpoints for PM2.5
(24-hour average, ug/m3), from the CPCB National Air Quality Index
report (2014). See README data source audit for links.

Category names follow CPCB: Good, Satisfactory, Moderately polluted,
Poor, Very poor, Severe.
"""

# (conc_low, conc_high, aqi_low, aqi_high, category)
PM25_BREAKPOINTS = [
    (0.0, 30.0, 0, 50, "Good"),
    (30.0, 60.0, 51, 100, "Satisfactory"),
    (60.0, 90.0, 101, 200, "Moderate"),
    (90.0, 120.0, 201, 300, "Poor"),
    (120.0, 250.0, 301, 400, "Very Poor"),
    (250.0, 500.0, 401, 500, "Severe"),
]


def pm25_to_aqi(pm25: float) -> int:
    """Convert a 24h mean PM2.5 (ug/m3) to the Indian AQI sub-index."""
    if pm25 is None or pm25 < 0:
        raise ValueError(f"invalid pm2.5 value: {pm25}")
    if pm25 >= 500.0:
        return 500
    for lo, hi, alo, ahi, _cat in PM25_BREAKPOINTS:
        if lo <= pm25 <= hi:
            return round(alo + (ahi - alo) * (pm25 - lo) / (hi - lo))
    # between bands due to float edges: clamp into the nearest band
    for lo, hi, alo, ahi, _cat in PM25_BREAKPOINTS:
        if pm25 < lo:
            return alo
    return 500


def aqi_category(aqi: int) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def pm25_category(pm25: float) -> str:
    return aqi_category(pm25_to_aqi(pm25))
