# Coffee Price Tracker & Forecast

A live coffee-futures data pipeline and dashboard, paired with an MRM-inspired model validation and monitoring layer: assumption testing, regime-segmented benchmarking, and GARCH-based anomaly detection.

🔗 Live dashboard: https://caphe-forecast.streamlit.app/ 

---
> Applies model-risk-management practices to a self-built model. Demonstrates validation discipline; not a regulated validation exercise.
---

## What it does

Ingests daily coffee futures, stores multi-year history in a hosted database, refreshes automatically, and serves a public dashboard — then he validation layer sits on top: testing whether the model's assumptions hold, benchmarking it against alternatives, and documenting where it's fragile with monitors on top of that pipeline.

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

## Architecture
 
```
yfinance (KC=F)
      │
      ▼
Python fetcher ──upsert──▶ Neon (PostgreSQL) ──▶ Streamlit dashboard ──▶ public URL
      ▲                          ▲
      │                          │
GitHub Actions (daily cron)   Validation & anomaly detection (offline)
```
Loads are **idempotent** (`INSERT ... ON CONFLICT`), so re-runs never duplicate
data. Backtests and validation tests run offline; the dashboard displays
evaluated results rather than recomputing them per page load.

## Tech stack
 
Python · PostgreSQL (Neon) · Streamlit + Plotly · GitHub Actions · `arch` (GARCH)
· statsmodels (ADF, ARCH tests) · numpy / pandas · yfinance

## Repository guide

| Path | Purpose |
|---|---|
| `scripts/fetch_futures.py` · `db.py` · `backfill.py` | Extract → idempotent load |
| `scripts/backtest.py` · `vol_backtest.py` · `regime_backtest.py` | Walk-forward benchmarking |
| `scripts/validate_assumptions.py` | ADF + ARCH-effect tests |
| `scripts/detect_anomalies.py` · `schema_anomalies.sql` | GARCH anomaly detection + storage |
| `scripts/check_vol_dates.py` | Regime dates & thresholds |
| `streamlit_app.py` | Dashboard (Price · Volatility · Monitoring) |
| `docs/validation_findings.md` | Full validation write-up |

## Running locally

```bash
git clone https://github.com/YOUR-USERNAME/caphe-forecast.git
cd caphe-forecast
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'DATABASE_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require' > .env
 
psql "$DATABASE_URL" -f scripts/schema.sql            # price table
psql "$DATABASE_URL" -f scripts/schema_anomalies.sql  # anomaly table
python scripts/backfill.py                             # load history
python scripts/detect_anomalies.py                     # compute anomaly flags
streamlit run streamlit_app.py
 
# reproduce the analysis
python scripts/validate_assumptions.py   # assumption tests
python scripts/regime_backtest.py        # regime-segmented scorecard
```
## Status & roadmap
 
**Live:** pipeline · daily refresh · volatility model · assumption testing ·
regime benchmarking · anomaly monitoring.
 
- [ ] Walk-forward (real-time) anomaly detector for the daily cron
- [ ] Anomaly attribution (price-gap vs. volatility-spike; isolated vs. persistent)
- [ ] Validation memo (formal write-up)
- [ ] Additional instruments (robusta, related softs) for cross-asset analysis

