from __future__ import annotations
 
import os
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st
 
SYMBOL = "KC=F"
SYMBOL_LABEL = "Arabica Coffee — ICE Coffee C (KC=F), US cents / lb"
 
FORECAST_HORIZON = 14   # trading days to project 
TREND_LOOKBACK = 30     # trading days used to estimate the trend 
 
# data 
def get_database_url() -> str:
    """Streamlit secrets when deployed; .env when local."""
    try:
        return st.secrets["DATABASE_URL"]
    except Exception:
        from dotenv import load_dotenv
 
        load_dotenv()
        url = os.environ.get("DATABASE_URL")
        if not url:
            st.error("DATABASE_URL not found in Streamlit secrets or .env")
            st.stop()
        return url
 
 
@st.cache_data(ttl=3600)
def load_prices(symbol: str) -> pd.DataFrame:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT obs_date, open, high, low, close, volume
                FROM market_data.prices
                WHERE symbol = %s
                ORDER BY obs_date
                """,
                (symbol,),
            )
            rows = cur.fetchall()
 
    df = pd.DataFrame(
        rows, columns=["obs_date", "open", "high", "low", "close", "volume"]
    )
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df
 
 
# this is the forecast part, just the simple linear trend model for now as the baseline, any improvements will be updated later and beat the baseline 
def get_forecast(df: pd.DataFrame, horizon: int = FORECAST_HORIZON,
                 lookback: int = TREND_LOOKBACK) -> pd.DataFrame:
    """
    Baseline forecast: fit a straight line to the last `lookback` closes and
    extend it `horizon` trading days forward. Simple and explainable.
 
    To upgrade the model later (ARIMA, Prophet, ML), change ONLY this function —
    it returns a DataFrame [obs_date, forecast] and the app doesn't care how the
    numbers were produced.
    """
    recent = df.tail(lookback)
    x = np.arange(len(recent))
    y = recent["close"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)  # degree-1 = straight-line trend
 
    future_x = np.arange(len(recent), len(recent) + horizon)
    future_y = slope * future_x + intercept
 
    last_date = df["obs_date"].iloc[-1]
    future_dates = pd.bdate_range(  # business days only 
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )
 
    # prepend the last actual point so the forecast line connects to history.
    fc = pd.DataFrame(
        {
            "obs_date": [last_date, *future_dates],
            "forecast": [df["close"].iloc[-1], *future_y],
        }
    )
    return fc
 
 
# signal 
def get_signal(df: pd.DataFrame, lookback: int = TREND_LOOKBACK) -> str:
    """One plain-language sentence describing the recent trend."""
    if len(df) <= lookback:
        return "Not enough history yet to assess a trend."
 
    now = df["close"].iloc[-1]
    then = df["close"].iloc[-lookback]
    pct = (now - then) / then * 100
 
    if pct > 2:
        trend = "trending upward"
    elif pct < -2:
        trend = "trending downward"
    else:
        trend = "roughly flat"
 
    return (
        f"Over the last {lookback} trading days, arabica has moved "
        f"{pct:+.1f}% — {trend}."
    )
 
 
#  page 
st.set_page_config(page_title="Coffee Price Dashboard", page_icon="☕", layout="wide")
st.title("☕ Coffee Price Dashboard")
st.caption(SYMBOL_LABEL)
st.markdown(
    "Tracking global arabica coffee futures. Shows recent price history, a "
    "simple trend forecast, and a plain-language market signal. "
    "**Indicative only — not trading advice.**"
)
 
df = load_prices(SYMBOL)
if df.empty:
    st.warning("No data found. Has the backfill been run?")
    st.stop()
 
# metrics
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
change = latest["close"] - prev["close"]
pct = (change / prev["close"] * 100) if prev["close"] else 0.0
 
col1, col2 = st.columns(2)
col1.metric(
    label=f"Latest close ({latest['obs_date'].date()})",
    value=f"{latest['close']:.2f}",
    delta=f"{change:+.2f} ({pct:+.2f}%)",
)
col2.metric(label="History", value=f"{len(df)} trading days")
 
# signal
st.info(get_signal(df))
 
# chart: history + forecast overlay
fc = get_forecast(df)
 
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["obs_date"], y=df["close"], name="History",
        line=dict(color="#6F4E37"),
    )
)
fig.add_trace(
    go.Scatter(
        x=fc["obs_date"], y=fc["forecast"], name=f"{FORECAST_HORIZON}-day forecast",
        line=dict(color="#C8964F", dash="dash"),
    )
)
fig.update_layout(
    height=460, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    yaxis_title="US cents / lb", xaxis_title=None,
)
# initial view: last ~6 months + the forecast, so the dashed line is visible
if len(df) > 126:
    fig.update_xaxes(range=[df["obs_date"].iloc[-126], fc["obs_date"].iloc[-1]])
st.plotly_chart(fig, use_container_width=True)
 
with st.expander("How the forecast works"):
    st.markdown(
        f"""
        The forecast is a **baseline**: a straight-line trend fitted to the last
        {TREND_LOOKBACK} trading days, extended {FORECAST_HORIZON} days forward.
        It's deliberately simple and explainable — a starting point to benchmark
        better models against, not a precise prediction. Daily futures prices are
        close to a random walk in the short term, so treat this as indicative of
        recent direction only.
        """
    )
 