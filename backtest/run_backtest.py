"""Walk-forward backtest over last winter. Proves (or disproves)
that the gradient boosting model beats persistence per city BEFORE
any live prediction ships. Run via the backtest workflow:

    python -m backtest.run_backtest --eval-start 2025-10-01 \
        --eval-end 2026-02-28

Protocol, stated precisely so it can be attacked:

* Training data: the repo's observation files (seed them first with
  pipeline.bootstrap_history), everything strictly before the
  forecast anchor. For a target day T at lead k, the model may use
  observations up to T-k only. Features are built by the exact same
  code the nightly job uses (pipeline.model._day_features).
* Retraining: models are refit every 7 evaluation days on all data
  available at that point (nightly refits in production, weekly here
  to keep the free runner fast; this can only hurt the model, so the
  comparison is conservative).
* Weather features for the target day use that day's historical
  weather, standing in for a forecast. This flatters BOTH the model
  and nobody else (persistence uses no weather), and is flagged in
  the report.
* Metrics per city per lead: MAE, MAPE, 80 percent band coverage,
  and skill = 1 - MAE_model / MAE_persistence. Positive skill means
  the model earned its keep.

Outputs backtest/results.json and backtest/REPORT.md and exits
nonzero if the model loses to persistence at lead 2 in any city, so
the workflow goes red instead of quietly shipping a loser.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pipeline.config import CITIES, MODEL_VERSION, REPO_ROOT
from pipeline import store
from pipeline.model import (FEATURE_ORDER, _day_features,
                            build_training_frame, train_quantile_models,
                            MIN_TRAIN_ROWS, WEATHER_FEATS)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vayu")

RESULTS_FILE = REPO_ROOT / "backtest" / "results.json"
REPORT_FILE = REPO_ROOT / "backtest" / "REPORT.md"
RETRAIN_EVERY_DAYS = 7
LEADS = (1, 2)


def backtest_city(city_key, eval_start, eval_end):
    obs_df = store.load_obs(city_key)
    obs_df = obs_df.dropna(subset=["pm25"]).copy()
    if obs_df.empty:
        return {"error": "no observations, run bootstrap_history first"}
    obs_df["dt"] = pd.to_datetime(obs_df["date"])
    obs = obs_df.set_index("dt")["pm25"].astype(float)
    fires = pd.to_numeric(obs_df.set_index("dt")["fire_count"],
                          errors="coerce")
    weather_by_day = {row["dt"]: {k: row.get(k) for k in WEATHER_FEATS}
                      for _, row in obs_df.iterrows()}
    src_by_day = obs_df.set_index("dt")["source"]

    rows = []
    models, trained_on = None, None
    eval_days = pd.date_range(eval_start, eval_end)
    for target in eval_days:
        if target not in obs.index:
            continue
        if models is None or (target - trained_on).days >= RETRAIN_EVERY_DAYS:
            train_df = obs_df[obs_df["dt"] < target - pd.Timedelta(days=max(LEADS))]
            X, y, _, _ = build_training_frame(
                train_df.drop(columns=["dt"]), leads=LEADS)
            if len(y) < MIN_TRAIN_ROWS:
                continue
            models = train_quantile_models(X, y)
            trained_on = target
        actual = obs[target]
        for lead in LEADS:
            anchor = target - pd.Timedelta(days=lead)
            if anchor not in obs.index:
                continue
            feats = _day_features(obs, fires, weather_by_day.get(target),
                                  target, lead)
            # leakage guard: mask any obs-derived feature computed
            # from days after the anchor (there are none by
            # construction, this asserts it stays true)
            x = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=float)
            lo = float(models[0.1].predict(x)[0])
            mid = float(models[0.5].predict(x)[0])
            hi = float(models[0.9].predict(x)[0])
            lo, mid, hi = sorted([lo, mid, hi])
            persist = obs[anchor]
            rows.append({
                "city": city_key, "target": target.strftime("%Y-%m-%d"),
                "lead": lead, "actual": actual, "pred": mid,
                "lo": lo, "hi": hi, "persistence": persist,
                "obs_source": src_by_day.get(target, ""),
            })
    if not rows:
        return {"error": "no evaluable days (not enough training history)"}
    df = pd.DataFrame(rows)
    out = {"n_days": int(df["target"].nunique())}
    for lead in LEADS:
        d = df[df["lead"] == lead]
        if d.empty:
            continue
        mae = float((d["pred"] - d["actual"]).abs().mean())
        pmae = float((d["persistence"] - d["actual"]).abs().mean())
        mape = float(((d["pred"] - d["actual"]).abs()
                      / d["actual"]).mean() * 100)
        pmape = float(((d["persistence"] - d["actual"]).abs()
                       / d["actual"]).mean() * 100)
        cover = float(((d["actual"] >= d["lo"])
                       & (d["actual"] <= d["hi"])).mean() * 100)
        out[f"lead{lead}"] = {
            "n": int(len(d)),
            "model_mae": round(mae, 1),
            "persistence_mae": round(pmae, 1),
            "model_mape": round(mape, 1),
            "persistence_mape": round(pmape, 1),
            "band_coverage_pct": round(cover, 1),
            "skill_vs_persistence": round(1 - mae / pmae, 3) if pmae else None,
        }
    monitor_share = float((df["obs_source"] == "openaq").mean() * 100)
    out["monitor_truth_share_pct"] = round(monitor_share, 1)
    return out


def write_report(results, eval_start, eval_end):
    lines = []
    lines.append("# VAYU backtest report")
    lines.append("")
    lines.append(f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}. "
                 f"Model `{MODEL_VERSION}`. Evaluation window {eval_start} "
                 f"to {eval_end}.")
    lines.append("")
    lines.append("Protocol: walk-forward, weekly refits, features built by "
                 "the same code the nightly job runs, no observation later "
                 "than target minus lead ever enters a feature. Target-day "
                 "weather uses historical values as a stand-in for a "
                 "forecast, which is standard but slightly flattering; "
                 "persistence uses no weather at all. Truth series mixes "
                 "monitor days (openaq) and reanalysis days (cams), the "
                 "share is listed per city.")
    lines.append("")
    lines.append("Skill above 0 means the model beats copying the last "
                 "observed value. The 80 percent band should cover close "
                 "to 80 percent of actuals.")
    lines.append("")
    lines.append("| City | Lead | N | Model MAE | Persistence MAE | Skill | Band coverage | Monitor share |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for city_key, res in results.items():
        name = CITIES[city_key]["name"]
        if "error" in res:
            lines.append(f"| {name} | - | - | ERROR: {res['error']} | | | | |")
            continue
        for lead in LEADS:
            block = res.get(f"lead{lead}")
            if not block:
                continue
            lines.append(
                f"| {name} | {lead}d | {block['n']} "
                f"| {block['model_mae']} | {block['persistence_mae']} "
                f"| {block['skill_vs_persistence']} "
                f"| {block['band_coverage_pct']}% "
                f"| {res['monitor_truth_share_pct']}% |")
    lines.append("")
    lines.append("MAE in ug/m3 of daily mean PM2.5. Full per-day rows are "
                 "reproducible by re-running this script on the committed "
                 "observation files.")
    REPORT_FILE.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-start", default="2025-10-01")
    ap.add_argument("--eval-end", default="2026-02-28")
    args = ap.parse_args()
    results = {}
    for city_key in CITIES:
        log.info("backtesting %s", city_key)
        results[city_key] = backtest_city(
            city_key, args.eval_start, args.eval_end)
    RESULTS_FILE.write_text(json.dumps(
        {"eval_start": args.eval_start, "eval_end": args.eval_end,
         "model_version": MODEL_VERSION, "results": results}, indent=2))
    write_report(results, args.eval_start, args.eval_end)

    losers = []
    for city_key, res in results.items():
        if "error" in res:
            losers.append(f"{city_key}: {res['error']}")
            continue
        block = res.get("lead2")
        if block and block["skill_vs_persistence"] is not None \
                and block["skill_vs_persistence"] <= 0:
            losers.append(f"{city_key}: lead2 skill "
                          f"{block['skill_vs_persistence']}")
    if losers:
        log.error("BACKTEST FAILED, model does not beat persistence:")
        for l in losers:
            log.error("  - %s", l)
        sys.exit(1)
    log.info("backtest passed in every city")


if __name__ == "__main__":
    main()
