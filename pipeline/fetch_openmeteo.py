"""Open-Meteo: weather features, forecast features, and the CAMS
air quality fallback series.

Free for non-commercial use, no key. Daily aggregation is done here
in IST because the model, the grading, and the cards all speak IST.
"""

import logging

import pandas as pd

from .config import (CITIES, IST, OPEN_METEO_AQ, OPEN_METEO_WEATHER,
                     WEATHER_HOURLY)
from .http_util import get_with_retries

log = logging.getLogger("vayu")


def _hourly_frame(payload, keys):
    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        raise RuntimeError("open-meteo returned no hourly data")
    df = pd.DataFrame({k: hourly.get(k) for k in ["time"] + keys})
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.strftime("%Y-%m-%d")
    return df


def cams_pm25_daily(city_key: str, past_days=7, forecast_days=3):
    """CAMS modeled PM2.5, daily mean by IST day. Used as the
    fallback observed series (flagged) and as a model feature."""
    city = CITIES[city_key]
    r = get_with_retries(OPEN_METEO_AQ, params={
        "latitude": city["lat"], "longitude": city["lon"],
        "hourly": "pm2_5",
        "past_days": str(past_days),
        "forecast_days": str(forecast_days),
        "timezone": IST,
    })
    df = _hourly_frame(r.json(), ["pm2_5"])
    daily = (df.dropna(subset=["pm2_5"])
               .groupby("date")["pm2_5"]
               .agg(["mean", "count"]))
    daily = daily[daily["count"] >= 18]
    return {d: round(float(v), 1) for d, v in daily["mean"].items()}


def cams_pm25_history(city_key: str, start_date: str, end_date: str):
    """Historical CAMS PM2.5 daily means via start_date/end_date.

    Open-Meteo serves reanalysis this way back to 2013. If the
    endpoint refuses the range, the caller falls back to past_days.
    """
    city = CITIES[city_key]
    r = get_with_retries(OPEN_METEO_AQ, params={
        "latitude": city["lat"], "longitude": city["lon"],
        "hourly": "pm2_5",
        "start_date": start_date, "end_date": end_date,
        "timezone": IST,
    })
    df = _hourly_frame(r.json(), ["pm2_5"])
    daily = (df.dropna(subset=["pm2_5"])
               .groupby("date")["pm2_5"]
               .agg(["mean", "count"]))
    daily = daily[daily["count"] >= 18]
    return {d: round(float(v), 1) for d, v in daily["mean"].items()}


def weather_daily(city_key: str, past_days=7, forecast_days=3):
    """Daily weather aggregates by IST day, past and forecast in one
    call. Returns {date: {feature: value}}."""
    city = CITIES[city_key]
    r = get_with_retries(OPEN_METEO_WEATHER, params={
        "latitude": city["lat"], "longitude": city["lon"],
        "hourly": ",".join(WEATHER_HOURLY),
        "past_days": str(past_days),
        "forecast_days": str(forecast_days),
        "timezone": IST,
    })
    df = _hourly_frame(r.json(), WEATHER_HOURLY)
    out = {}
    for date, g in df.groupby("date"):
        row = {}
        row["wind_speed_mean"] = g["wind_speed_10m"].mean()
        row["wind_speed_min"] = g["wind_speed_10m"].min()
        row["temp_mean"] = g["temperature_2m"].mean()
        row["temp_min"] = g["temperature_2m"].min()
        row["rh_mean"] = g["relative_humidity_2m"].mean()
        row["pressure_mean"] = g["surface_pressure"].mean()
        row["precip_sum"] = g["precipitation"].sum()
        blh = g["boundary_layer_height"]
        row["blh_min"] = blh.min() if blh.notna().any() else None
        row["blh_mean"] = blh.mean() if blh.notna().any() else None
        out[date] = {k: (round(float(v), 2) if v is not None and
                         pd.notna(v) else None)
                     for k, v in row.items()}
    return out


def weather_history(city_key: str, start_date: str, end_date: str):
    """Daily weather aggregates from the ERA5 archive endpoint for
    backtesting. Note: archive has no boundary_layer_height, so the
    backtest and the live model both treat BLH as optional."""
    from .config import OPEN_METEO_ARCHIVE
    city = CITIES[city_key]
    hourly = [h for h in WEATHER_HOURLY if h != "boundary_layer_height"]
    r = get_with_retries(OPEN_METEO_ARCHIVE, params={
        "latitude": city["lat"], "longitude": city["lon"],
        "hourly": ",".join(hourly),
        "start_date": start_date, "end_date": end_date,
        "timezone": IST,
    }, timeout=60)
    df = _hourly_frame(r.json(), hourly)
    out = {}
    for date, g in df.groupby("date"):
        out[date] = {
            "wind_speed_mean": round(float(g["wind_speed_10m"].mean()), 2),
            "wind_speed_min": round(float(g["wind_speed_10m"].min()), 2),
            "temp_mean": round(float(g["temperature_2m"].mean()), 2),
            "temp_min": round(float(g["temperature_2m"].min()), 2),
            "rh_mean": round(float(g["relative_humidity_2m"].mean()), 2),
            "pressure_mean": round(float(g["surface_pressure"].mean()), 2),
            "precip_sum": round(float(g["precipitation"].sum()), 2),
            "blh_min": None,
            "blh_mean": None,
        }
    return out
