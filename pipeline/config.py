"""VAYU configuration: cities, endpoints, paths.

All data lives in flat files inside the repo. No database.
Timezone for all daily aggregation is IST (Asia/Kolkata).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OBS_DIR = DATA_DIR / "observations"
PRED_FILE = DATA_DIR / "predictions" / "predictions.csv"
SCORES_FILE = DATA_DIR / "scores" / "scores.csv"
SCOREBOARD_FILE = DATA_DIR / "scoreboard.json"
LATEST_FILE = DATA_DIR / "latest.json"
COMPARISONS_FILE = DATA_DIR / "comparisons.json"
RUNLOG_FILE = DATA_DIR / "runlog.json"
CARDS_OUT = REPO_ROOT / "out" / "cards"

IST = "Asia/Kolkata"

# Season window used for the public scoreboard (rolling, defined in README)
SEASON_WINDOW_DAYS = 90

# Minimum hourly values for a station-day to count as observed (75% of 24)
MIN_HOURS_PER_DAY = 18

MODEL_VERSION = "gbq-1.0"

CITIES = {
    "delhi": {
        "name": "Delhi",
        "lat": 28.6139,
        "lon": 77.2090,
        "cpcb_city": "Delhi",
        # Punjab + Haryana bounding box (west, south, east, north) for
        # stubble fire counts feeding the Delhi model (Oct and Nov mainly)
        "fires_bbox": "73.5,27.5,77.7,32.7",
        "openaq_radius_m": 25000,
    },
    "mumbai": {
        "name": "Mumbai",
        "lat": 19.0760,
        "lon": 72.8777,
        "cpcb_city": "Mumbai",
        "fires_bbox": None,
        "openaq_radius_m": 25000,
    },
    "pune": {
        "name": "Pune",
        "lat": 18.5204,
        "lon": 73.8567,
        "cpcb_city": "Pune",
        # Western Maharashtra box. Crop residue and waste fires exist here
        # but matter far less than Punjab fires do for Delhi. The model
        # decides the weight; we just supply the count.
        "fires_bbox": "72.6,17.0,76.5,21.5",
        "openaq_radius_m": 25000,
    },
    "bengaluru": {
        "name": "Bengaluru",
        "lat": 12.9716,
        "lon": 77.5946,
        "cpcb_city": "Bengaluru",
        "fires_bbox": None,
        "openaq_radius_m": 25000,
    },
    "nashik": {
        "name": "Nashik",
        "lat": 19.9975,
        "lon": 73.7898,
        "cpcb_city": "Nashik",
        "fires_bbox": "72.6,17.0,76.5,21.5",
        "openaq_radius_m": 25000,
    },
}

# Endpoints
OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# CPCB real time AQI via data.gov.in
# Resource: "Real time Air Quality Index from various locations"
CPCB_RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_URLS = [
    f"https://api.data.gov.in/resource/{CPCB_RESOURCE}",
]

OPENAQ_BASE = "https://api.openaq.org/v3"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Weather features pulled for the model (hourly, aggregated to daily)
WEATHER_HOURLY = [
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",
]
