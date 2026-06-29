# ☕ Coffee Price Tracker & Forecast
A deployed web dashboard tracking global arabica coffee futures (ICE Coffee C),
with price history, a simple forecast, and a plain-language market signal. 
 🔗 Live demo: https://caphe-forecast.streamlit.app/
 
## Status
Live and self-maintaining. The full data pipeline, hosted database, deployed
dashboard, and daily automated refresh are all built and running. The forecasting
analysis is complete.

## The Project
An end-to-end data pipeline and dashboard for coffee futures prices. It ingests
daily market data, stores a multi-year history in a hosted PostgreSQL database,
refreshes itself daily, and serves everything through a public Streamlit app.

It tracks the global benchmark (arabica futures), useful as a market
indicator — not personalized trading advice.

## Features:
- Latest price with day-over-day change
- Interactive price-history chart (Plotly)
- Plain-language signal summarizing the recent trend
- Volatility analysis — rolling volatility, current regime, and model results
- Automatic daily data refresh — no manual intervention

## Forecasting

The forecasting work produced two findings, both validated with **walk-forward
backtesting** (train on the past, predict the future, roll forward — never
letting future data leak into a prediction):

**1. Price level is close to a random walk.** Trend extrapolation and drift
models both *lost* to a naive "tomorrow = today" baseline:

| Model            | RMSE  | vs. baseline   |
|------------------|-------|----------------|
| naive (baseline) | 17.66 | —              |
| drift            | 19.86 | +12.5% worse   |
| trend            | 25.33 | +43.4% worse   |

This is consistent with weak-form market efficiency — short-term price direction
carries little memory.

**2. Volatility *is* forecastable**, because it clusters (calm follows calm,
turbulent follows turbulent). Predicting realized volatility, both models beat
the persistence baseline:

| Model                  | RMSE     | vs. baseline   |
|------------------------|----------|----------------|
| persistence (baseline) | 0.00559  | —              |
| EWMA                   | 0.00486  | −13.1% better  |
| GARCH(1,1)             | 0.00379  | −32.1% better  |

**Takeaway:** direction has no memory, but magnitude does. The dashboard shows
the price forecast as an honest baseline (flagged as indicative), and surfaces
the volatility analysis — the genuinely predictable signal — in its own tab.

## Tech stack

| Layer        | Tool                                  |
|--------------|---------------------------------------|
| Data source  | yfinance (`KC=F`, ICE Coffee C)       |
| Database     | Neon (serverless PostgreSQL)          |
| Modelling    | numpy, EWMA, GARCH (`arch`)           |
| App / UI     | Streamlit + Plotly                    |
| DB driver    | psycopg                               |
| Automation   | GitHub Actions (daily cron)           |
| Hosting      | Streamlit Community Cloud             |

## Project notes

- **Data-source pivot.** The original plan scraped Vietnamese farmgate price
  aggregators. Source reconnaissance found the candidates were duplicates,
  stale, or actively defended with CSS obfuscation — so the project pivoted to a
  clean, reliable futures API rather than maintain a brittle scraper.
- **Schema design.** A natural composite key (`obs_date`, `symbol`, `source`)
  serves as the primary key, making the daily upsert idempotent.
- **One config, three environments.** Database credentials are read from a local
  `.env` (dev), Streamlit secrets (app), and a GitHub Actions secret (cron) — the
  same code runs in all three.
- **Precomputed evaluation.** Backtests (including refit-per-window GARCH) run
  offline; the app displays the evaluated results, so page loads stay fast and
  the heavy modelling dependency stays out of the deployment.

## Running locally

```bash
git clone https://github.com/YOUR-USERNAME/caphe-forecast.git
cd caphe-forecast
python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
echo 'DATABASE_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require' > .env
psql "$DATABASE_URL" -f scripts/schema.sql           # create the table
python scripts/backfill.py                            # backfill history
streamlit run streamlit_app.py                        # run the app

# reproduce the analysis:
python scripts/backtest.py        # price-level backtest
pip install arch
python scripts/vol_backtest.py    # volatility backtest (persistence / EWMA / GARCH)
```

> `.env` is gitignored and never committed.

## Roadmap

- [ ] Add robusta as a second instrument
- [ ] External features (weather, FX) for a returns/direction model
- [ ] Date-range controls and richer chart interactions

