# Coffee Price Tracker & Forecast

A production-style forecasting pipeline for coffee futures, paired with an MRM-inspired model validation layer: assumption testing, multi-baseline benchmarking, and structural-break analysis. 

> This project applies model-risk-management *practices* (assumption testing,
> benchmarking, limitations documentation) to a self-built forecasting model. It
> demonstrates validation discipline; it is not a regulated validation exercise.

---
> Applies model-risk-management *practices* to a self-built forecasting model.
> It demonstrates validation discipline; it is not a regulated validation exercise.
---

## Why this project

This one asks the question a model validator asks: **should this model be trusted, and where does
it break?**

Pipeline: it ingests data daily, stores it in a hosted database,
and serves a public dashboard. The validation layer sits on top: testing whether
the model's assumptions hold, benchmarking it against alternatives, and
documenting where it's fragile.

## Headline findings

_Data through 2026-07-21 · run 2026-07-22 · all figures from walk-forward
backtesting (no look-ahead bias)_

- **Price levels are not forecastable.** Trend and drift models both lose to a
  naive baseline (+43.4% and +12.5% RMSE). Consistent with weak-form efficiency —
  this is what motivated re-targeting the model at volatility.
- **Volatility is forecastable**, because it clusters. GARCH(1,1) beats a
  persistence baseline by **−28.8% RMSE**, ahead of EWMA (−12.3%) and a 30-day
  historical estimate (−4.3%).
- **The model's edge widens under stress**, contrary to the initial concern:
  −21.5% in calm regimes → −31.8% normal → **−39.1% turbulent**.
- **Assumptions verified, not assumed.** Returns are stationary (ADF p ≈ 0) and
  ARCH effects are overwhelming (Engle LM p ≈ 7 × 10⁻¹⁸), so a GARCH-class model
  is an evidenced choice.
- **Principal open risk:** out-of-distribution regimes. Volatility currently sits
  at the **99.9th percentile** of the sample — above anything in the fitted
  history — where GARCH's mean-reversion anchor may cause under-forecasting. The
  backtest cannot measure this; live monitoring is the recommended next control.

→ Methodology, full tables, and caveats in
[`notebooks/validation_findings.md`](notebooks/validation_findings.md).

## Status

| Component | Status |
|---|---|
| Data pipeline (ingest → Postgres → dashboard) | Live |
| Daily automated refresh (GitHub Actions) | Live |
| Volatility model (GARCH / EWMA) | Complete |
| Assumption testing (ADF, ARCH effects) | Complete |
| Regime-segmented benchmarking |  Complete |
| Structural-break analysis | Complete |
| Ongoing monitoring (drift tracking) | In progress |
| Validation memo | Planned |

## Architecture

```
yfinance (KC=F)
      │
      ▼
Python fetcher ──upsert──▶ Neon (hosted PostgreSQL)
      ▲                          │
      │                          ▼
GitHub Actions (daily cron)   Streamlit dashboard ──▶ public URL
                                   │
                                   ▼
                        Validation layer (offline):
                        assumptions · benchmarking · breaks
```

Loads are **idempotent** (`INSERT ... ON CONFLICT`), so re-runs never duplicate
data. Backtests and validation tests run offline; the dashboard displays
evaluated results rather than recomputing them per page load.

## Tech stack

| Layer | Tool |
|---|---|
| Data source | yfinance (`KC=F`, ICE Coffee C) |
| Database | Neon (serverless PostgreSQL) |
| Modelling | `arch` (GARCH), numpy (EWMA) |
| Validation | statsmodels (ADF, Engle LM, Ljung–Box) |
| App / UI | Streamlit + Plotly |
| Automation | GitHub Actions (daily cron) |
| Hosting | Streamlit Community Cloud |

## Repository guide

| Path | Purpose |
|---|---|
| `scripts/fetch_futures.py` | Extract daily OHLCV from the API |
| `scripts/db.py` | Idempotent upsert layer |
| `scripts/backfill.py` | Full historical load / daily refresh |
| `scripts/backtest.py` | Walk-forward backtest — price models |
| `scripts/vol_backtest.py` | Walk-forward backtest — volatility models |
| `scripts/regime_backtest.py` | Regime-segmented scorecard |
| `scripts/validate_assumptions.py` | ADF, ARCH-effect tests, break plot |
| `scripts/check_vol_dates.py` | Maps volatility regimes to calendar dates |
| `streamlit_app.py` | Dashboard (price + volatility analysis) |
| `docs/validation_findings.md` | Full validation write-up |

## Running locally

```bash
git clone https://github.com/YOUR-USERNAME/caphe-forecast.git
cd caphe-forecast
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'DATABASE_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require' > .env
psql "$DATABASE_URL" -f scripts/schema.sql
python scripts/backfill.py
streamlit run streamlit_app.py

# reproduce the validation work
python scripts/validate_assumptions.py  
python scripts/regime_backtest.py        
python scripts/check_vol_dates.py        
```

## Roadmap

- [ ] **Ongoing monitoring** — rolling forecast error with drift thresholds,
      surfaced on the dashboard
- [ ] **Validation memo** — formal write-up: background, assumptions,
      benchmarking, findings, recommendations
- [ ] Verify causal attribution for the 2021 and 2026 volatility episodes
- [ ] Additional instruments (robusta, related softs) for cross-asset analysis

