"""Render one 1080x1920 PNG per city from data/latest.json.

    python -m cards.render_cards

Runs inside the nightly Action after the pipeline. Pure local
rendering: vendored fonts, inline grain, no network. Output lands in
out/cards/{date}/{city}.png plus a stable copy at
out/cards/latest/{city}.png so the site and Shama's phone always
have one predictable URL per city.
"""

import json
import logging
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CARDS_OUT, CITIES, LATEST_FILE  # noqa: E402
from pipeline.aqi import pm25_to_aqi, aqi_category  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vayu")

TEMPLATE = Path(__file__).parent / "template.html"

CATEGORY_COLORS = {
    "Good": "#3a7d74",
    "Satisfactory": "#3a7d74",
    "Moderate": "#c9a84c",
    "Poor": "#b85c38",
    "Very Poor": "#b85c38",
    "Severe": "#b85c38",
}


def month_word(date_iso):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    y, m, d = date_iso.split("-")
    return f"{int(d):02d} {months[int(m) - 1]} {y}"


def emphasize_numbers(text):
    """Wrap digit groups in <em> so the comparison line pops."""
    import re
    return re.sub(r"(\d[\d.,]*x?|\d+ percent)", r"<em>\1</em>", text)


def build_html(city_key, payload, season_acc):
    entry = payload["cities"][city_key]
    html = TEMPLATE.read_text()

    now = entry.get("now") or {}
    today_pred = entry.get("today") or {}
    now_pm = now.get("pm25") or today_pred.get("pm25")
    if now_pm is None:
        raise RuntimeError(f"{city_key}: no current value and no "
                           "today prediction, card would be empty")
    now_aqi = pm25_to_aqi(now_pm)
    now_cat = aqi_category(now_aqi)
    now_label = ("measured right now" if now.get("source") == "cpcb"
                 else "predicted for today")

    tm = entry.get("tomorrow") or {}
    tm_pm = tm.get("pm25")
    tm_cat = tm.get("category", "")

    comp = entry.get("comparison") or {}
    raw_comp = comp.get("text", "")
    comp_text = emphasize_numbers(raw_comp)
    comp_class = ("verylong" if len(raw_comp) > 170
                  else "long" if len(raw_comp) > 110 else "")
    comp_src = comp.get("source_name", "")
    if len(comp_src) > 60:
        comp_src = comp_src[:57] + "..."

    y = entry.get("yesterday")
    if y and y.get("pct_error") is not None:
        hit = "inside" if y["in_band"] else "outside"
        receipt = (f"Predicted <strong>{y['predicted']:g}</strong>, "
                   f"actual <strong>{y['actual']:g}</strong>. "
                   f"Off by {y['pct_error']:g}%, {hit} our band.")
    else:
        receipt = "First predictions are in the ledger. Grading starts tomorrow."
    if season_acc is not None:
        season_line = (f"season accuracy {season_acc:g}% | "
                       "method + receipts: link in bio")
    else:
        season_line = "scoreboard goes live after the first graded week"

    reps = {
        "{{DATE_LINE}}": month_word(payload["date_ist"]),
        "{{CITY}}": entry["name"],
        "{{NOW_LABEL}}": now_label,
        "{{NOW_PM25}}": f"{now_pm:g}",
        "{{NOW_CATEGORY}}": now_cat,
        "{{NOW_AQI}}": str(now_aqi),
        "{{NOW_COLOR}}": CATEGORY_COLORS.get(now_cat, "#c9a84c"),
        "{{TM_PM25}}": f"{tm_pm:g}" if tm_pm is not None else "?",
        "{{TM_LO}}": f"{tm.get('lo', 0):g}",
        "{{TM_HI}}": f"{tm.get('hi', 0):g}",
        "{{TM_CATEGORY}}": tm_cat,
        "{{TM_COLOR}}": CATEGORY_COLORS.get(tm_cat, "#c9a84c"),
        "{{COMPARISON}}": comp_text or "Air is invisible. Numbers are not.",
        "{{COMPARISON_CLASS}}": comp_class,
        "{{COMPARISON_SOURCE}}": comp_src or "VAYU",
        "{{RECEIPT_LINE}}": receipt,
        "{{SEASON_LINE}}": season_line,
        "{{HANDLE}}": "sovereign by source",
    }
    for k, v in reps.items():
        html = html.replace(k, str(v))
    return html


KEEP_DATED_DAYS = 90


def prune_old_cards():
    """Keep the repo lean: dated card folders older than 90 days are
    deleted. Their numbers stay forever in the CSV ledger and git
    history; only the heavy PNGs are pruned."""
    import re
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=KEEP_DATED_DAYS)).isoformat()
    if not CARDS_OUT.exists():
        return
    for child in CARDS_OUT.iterdir():
        if child.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", child.name):
            if child.name < cutoff:
                shutil.rmtree(child)
                log.info("pruned old cards: %s", child.name)


def main():
    prune_old_cards()
    payload = json.loads(LATEST_FILE.read_text())
    scoreboard = {}
    sb_file = LATEST_FILE.parent / "scoreboard.json"
    if sb_file.exists():
        scoreboard = json.loads(sb_file.read_text())

    date_dir = CARDS_OUT / payload["date_ist"]
    latest_dir = CARDS_OUT / "latest"
    date_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1080, "height": 1920})
        for city_key in CITIES:
            if city_key not in payload["cities"]:
                failures.append(f"{city_key} missing from latest.json")
                continue
            try:
                season = (scoreboard.get("cities", {})
                          .get(city_key, {})
                          .get("season", {}) or {}).get("lead2") or {}
                html = build_html(city_key, payload,
                                  season.get("accuracy"))
                tmp = TEMPLATE.parent / f"_render_{city_key}.html"
                tmp.write_text(html)
                page.goto(f"file://{tmp}")
                page.wait_for_timeout(400)  # font settle
                out = date_dir / f"{city_key}.png"
                page.screenshot(path=str(out))
                shutil.copy(out, latest_dir / f"{city_key}.png")
                tmp.unlink()
                log.info("card rendered: %s", out)
            except Exception as e:
                failures.append(f"{city_key}: {e}")
        browser.close()
    if failures:
        for f in failures:
            log.error("CARD FAILED: %s", f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
