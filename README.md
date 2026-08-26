# VAYU

Autonomous air quality forecasts for five Indian cities, with receipts.

Every night at about 03:15 IST, a GitHub Action wakes up, fetches fresh data,
grades every prediction that has matured against what actually happened,
retrains, publishes the next 48 hours with an uncertainty band, renders a
shareable card per city, and commits everything back to this repository.
No server, no database, no manual steps. The repo is the database, and
because every prediction lands as a timestamped git commit before the
outcome is known, the accountability is verifiable by anyone with a browser.

Cities: Delhi, Mumbai, Pune, Bengaluru, Nashik.

A [Sovereign by Source](https://shamathakur.carrd.co) project.

## How it works

```
GitHub Actions, nightly 21:45 UTC (03:15 IST)
   |
   1. update observations (last 7 days, self-healing backfill)
   |    truth: OpenAQ monitor daily medians -> fallback: CAMS model, flagged
   |    features: Open-Meteo weather, NASA FIRMS fire counts
   2. grade every matured prediction, refresh scoreboard.json
   3. train per-city gradient boosting (quantile 0.1 / 0.5 / 0.9)
   |    floor: persistence baseline, always computed alongside
   4. predict today + tomorrow per city, append to the ledger
   5. render 1080x1920 cards (Playwright screenshot of an HTML template)
   6. commit data/ and out/ -> the static site redeploys itself
   |
   any failure -> run goes red, GitHub emails the owner,
                  next night's backfill repairs the gaps
```

Flat files only:

```
data/observations/{city}.csv    one row per IST day: PM2.5, source, weather, fires
data/predictions/predictions.csv  append-only ledger, never rewritten
data/scores/scores.csv          one row per graded prediction
data/scoreboard.json            current rolling stats per city
data/latest.json                tonight's card payloads
data/comparisons.json           the sourced comparison library
out/cards/latest/{city}.png     tonight's share card per city
out/cards/{date}/{city}.png     dated cards, pruned after 90 days
backtest/REPORT.md              walk-forward proof vs the naive baseline
```

## The scoreboard, defined precisely

So nobody can accuse this project of cherry-picking, the published headline
number is defined here, once, and computed by `pipeline/grade.py` with no
exceptions and no exclusions:

**Season accuracy** = 100 minus the mean absolute percentage error (MAPE)
of ALL graded 48 hour predictions (lead 2) whose target dates fall in the
trailing 90 days, floored at zero.

Where:

* A prediction is the median (q50) forecast of the daily mean PM2.5 for an
  IST calendar day, made the night before (lead 1, "today") or two nights
  before (lead 2, "tomorrow"). Lead 2 is the headline. Lead 1 is also
  published.
* The observed truth for a city and day is the MEDIAN across that city's
  PM2.5 monitors (via OpenAQ, within 25 km of the city point) of each
  monitor's daily mean, counting only monitors with at least 75 percent
  daily coverage. If no monitor qualifies that day, the value falls back to
  the Open-Meteo CAMS model daily mean and the row is flagged
  `source=cams` in public. Model-graded days are never hidden.
* Every graded prediction appears in `data/scores/scores.csv` with its
  absolute error, percentage error, the persistence baseline's error on the
  same day, and whether the actual fell inside our published band.
* Alongside accuracy we always publish: MAE, the persistence baseline MAE
  (if we cannot beat copying yesterday's number, you will see it), the 80
  percent band hit rate (target: 80 percent), and the count of graded
  predictions.

Predictions are never edited after being committed. Grading rows are append
only. Git history is the audit trail.

## The model, honestly

* Baseline: persistence. The forecast for day T at lead k is the observed
  value at day T minus k. Free, brutal, surprisingly hard to beat.
* Candidate: scikit-learn HistGradientBoostingRegressor with quantile loss
  (0.1, 0.5, 0.9), one model set per city, retrained nightly on the city's
  own history. Features: recent PM2.5 lags and rolling means, day-of-year
  seasonality, forecast weather for the target day (wind, temperature,
  humidity, pressure, rain, boundary layer height), and satellite fire
  counts for the cities with an upwind burning problem (Delhi: Punjab and
  Haryana box; Pune and Nashik: a western Maharashtra box, where the
  effect is expected to be much smaller and the model is free to ignore it).
* The 80 percent band comes from the quantile models, then a calibration
  guard widens it (never narrows) to at least the 10th to 90th percentile
  of the last 60 graded residuals for that city and lead.
* A city with under 120 usable training rows falls back to persistence
  plus an error-quantile band, and the ledger says so in `model_version`.
* Before the first live prediction, run the backtest workflow. It replays
  last winter walk-forward, using only information available at forecast
  time, and fails the workflow if the model loses to persistence at lead 2
  in any city. The committed `backtest/REPORT.md` is the proof, including
  the honest caveats (weekly refits, historical weather standing in for
  forecast weather, and the monitor vs model truth share per city).

## Data source audit

Verified working on 2026-08-25. Every wobble found is listed. Nothing is
silently skipped at runtime: every failure is logged, emailed, and repaired
by the next night's backfill where possible.

| Source | Status today | Key | Used for | Limits and gaps |
|---|---|---|---|---|
| CPCB real time AQI via [data.gov.in](https://data.gov.in) (resource `3b01bcb8`) | Working, live station data | Free signup. Ships with the public sample key as fallback, which is heavily throttled | The "measured right now" number on cards | Current snapshot only, no history at this endpoint. Known flaky: the fetcher tries query and header auth, retries with backoff, and falls back to OpenAQ or the latest daily value. Nashik station coverage is thin; on gap days the card says "predicted for today" instead |
| [OpenAQ v3](https://docs.openaq.org) | Working | Free API key required | The graded daily truth (monitor medians) and all monitor history | Free tier rate limits apply (see their limits page). VAYU stays tiny: sensor discovery is cached, a normal night is well under 100 requests. OpenAQ mirrors CPCB stations; if CPCB feeds go dark upstream, monitor days go to the CAMS fallback and are flagged |
| [Open-Meteo](https://open-meteo.com) air quality + weather + archive | Working | None | Weather features, forecasts, CAMS PM2.5 fallback series, ERA5 backtest weather | Free for non-commercial use. Their terms: "Less than 10'000 API calls per day, 5'000 per hour and 600 per minute." VAYU uses about 40 calls a night. The air quality endpoint documents `past_days` up to 92; the bootstrap requests older history with `start_date`, which their docs describe less clearly, so the bootstrap logs loudly if a range is refused and the OpenAQ monitor history carries the seed instead |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) area API | Working | Free MAP_KEY signup | Active fire counts (VIIRS): stubble season feature for Delhi, minor feature for Pune and Nashik | 5000 transactions per 10 minutes, max 5 days per query. VAYU uses a handful of calls a night. Fire counts are recorded as missing (not zero) when the key is absent or the call fails, so the data never lies about what was looked at |
| GitHub Actions | n/a | n/a | The entire pipeline | Free for public repositories on standard runners. The nightly run takes a few minutes. Keep the repo public and this stays free forever |
| GitHub Pages | n/a | n/a | The site | Free, soft limits (100 GB bandwidth per month, 1 GB site). Dated card PNGs are pruned after 90 days to keep the repo lean; the numbers behind them stay forever in the CSVs |

Honest gaps to know about:

* CPCB's own portal (cpcb.nic.in / airquality.cpcb.gov.in) publishes
  richer data but has no stable public API and blocks robots; VAYU does
  not scrape it. The data.gov.in resource is the official programmatic
  door, and it is moody. That is why OpenAQ carries the truth series.
* Historical fire counts for the backtest use the VIIRS standard
  processing archive (`VIIRS_SNPP_SP`); very recent days use near real
  time (`VIIRS_SNPP_NRT`). Counts can differ slightly between the two.
* Nashik has the fewest monitors of the five cities. Expect more
  `source=cams` days there. The scoreboard's `obs_source` column makes
  those days visible instead of pretending otherwise.
* The Indian AQI shown is the PM2.5 sub-index. The official city AQI
  takes the worst sub-index across pollutants, so on days when another
  pollutant dominates, the official AQI can read higher.

## Cards

One 1080x1920 PNG per city per night, written to
`out/cards/latest/{city}.png` (stable URL, always tonight's card) and
`out/cards/{date}/{city}.png`. Grab them from your phone via the site or
directly from GitHub. Card hierarchy: current number, tomorrow's call with
its band, one comparison from the library, yesterday's receipt.

Every comparison in `data/comparisons.json` carries its exact conversion
math, a primary source URL, and an honest caveat. The rotation is
deterministic by date, not random. If a comparison cannot be computed
honestly for the day's value (for example pack math on a clean day), it is
skipped for that day. Unsourced numbers do not ship.

## Setup from your phone, once

Everything below works in a mobile browser. About 20 minutes, mostly
waiting for two workflow runs.

**1. Get three free API keys** (no credit card anywhere):

* data.gov.in: register at data.gov.in, copy your API key from your
  profile. (Skippable at first: the bundled sample key works, throttled.)
* OpenAQ: sign up at explore.openaq.org, copy the API key from settings.
* NASA FIRMS: request a MAP_KEY at firms.modaps.eosdis.nasa.gov/api/area/
  (the Get MAP_KEY link), it arrives by email.

**2. Add them as repository secrets.** Repo Settings, then Secrets and
variables, then Actions, then New repository secret, three times:

* `DATA_GOV_IN_KEY`
* `OPENAQ_KEY`
* `FIRMS_KEY`

**3. Turn on the site.** Repo Settings, then Pages, then under Source pick
Deploy from a branch, branch `main`, folder `/ (root)`. The site appears at
`https://<username>.github.io/<repo>/` a minute later. (Prefer Vercel?
Import the repo there instead; it is static files, nothing to configure.)

**4. Seed history.** Actions tab, choose `bootstrap-history`, Run
workflow. Takes a few minutes and commits the observation files.

**5. Prove the model.** Actions tab, choose `backtest`, Run workflow. When
it finishes green, read `backtest/REPORT.md` in the repo. If it goes red,
the model lost to the naive baseline somewhere and should not ship; open an
issue with the report and fix before going live.

**6. First live run.** Actions tab, choose `nightly-forecast`, Run
workflow. When it is green: site has cards, ledger has its first
predictions. From tomorrow it runs itself at about 03:15 IST.

**Daily routine after that:** open the site, save tonight's card, post it.
That is the entire job. If a night fails, GitHub emails you and the next
night backfills automatically; a red run needs no action from you unless it
stays red for several days.

**Manual run any time:** Actions, nightly-forecast, Run workflow.

## Repo layout

```
pipeline/        fetchers, model, grading, nightly orchestrator
backtest/        walk-forward evaluation, report committed here
cards/           HTML template + Playwright renderer + vendored fonts
data/            observations, ledger, scores, scoreboard, comparisons
out/cards/       rendered PNGs
index.html       the site (static, reads the JSON and CSV in this repo)
.github/         nightly cron + bootstrap + backtest workflows
```

## License

MIT. Data credits: CPCB via data.gov.in, OpenAQ and its providers,
Open-Meteo (non-commercial), NASA FIRMS. Comparison sources are cited
inline in `data/comparisons.json` and on every card.
