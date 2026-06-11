# ☕ Coffee Price Tracker & Forecast
A deployed web dashboard tracking global arabica coffee futures (ICE Coffee C),
with price history, a simple forecast, and a plain-language market signal. 
 🔗 Live demo: https://caphe-forecast.streamlit.app/
## Status
Live and self-maintaining. The full data pipeline, hosted database, deployed
dashboard, and daily automated refresh are all built and running. A simple
baseline forecast is live in the app.

🚧 In progress: a proper forecasting model with walk-forward backtesting,
benchmarked against a naive baseline with honest error metrics. The current
dashboard forecast is a transparent baseline while this work is underway.

## The Project
An end-to-end data pipeline and dashboard for coffee futures prices. It ingests
daily market data, stores a multi-year history in a hosted PostgreSQL database,
refreshes itself daily, and serves everything through a public Streamlit app.

It tracks the global benchmark (arabica futures), useful as a market
indicator — not personalized trading advice.

## Forecast
The current forecast is a baseline: a straight-line trend fitted to recent
closing prices and extended forward. It is deliberately simple and explainable.

Daily futures prices behave close to a random walk in the short term, so this is
indicative of recent direction only. The baseline exists primarily as a
benchmark — the in-progress work is a more rigorous model evaluated with
walk-forward backtesting and compared against this baseline using honest
error metrics (MAE/RMSE) (expected) 

## Roadmap
Improved forecast — model with walk-forward backtesting, benchmarked
against the baseline with honest error metrics (in progress)
Add robusta as a second instrument
Date-range controls and richer chart interactions

