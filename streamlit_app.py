from __future__ import annotations
 
import os
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
SYMBOL = "KC=F"
SYMBOL_LABEL = "Arabica Coffee — ICE Coffee C (KC=F), US cents / lb"
FORECAST_HORIZON = 14
TREND_LOOKBACK = 30
VOL_WINDOW = 21
 
# Pre-computed walk-forward backtest results (from scripts/*.py). See the
# repo / notebook for the methodology behind these numbers.
PRICE_BACKTEST = pd.DataFrame([
    {"model": "naive (baseline)", "RMSE": 17.66, "vs baseline": "—"},
    {"model": "drift",            "RMSE": 19.86, "vs baseline": "+12.5% worse"},
    {"model": "trend",            "RMSE": 25.33, "vs baseline": "+43.4% worse"},
])
VOL_BACKTEST = pd.DataFrame([
    {"model": "persistence (baseline)", "RMSE": 0.00559, "vs baseline": "—"},
    {"model": "EWMA",                   "RMSE": 0.00486, "vs baseline": "-13.1% better"},
    {"model": "GARCH(1,1)",             "RMSE": 0.00379, "vs baseline": "-32.1% better"},
])
 
 
# ---------------- data ----------------
def get_database_url() -> str:
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
    import psycopg
    with psycopg.connect(get_database_url()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT obs_date, close FROM market_data.prices "
            "WHERE symbol = %s ORDER BY obs_date",
            (symbol,),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["obs_date", "close"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["close"] = df["close"].astype(float)
    # volatility columns (cheap to compute live)
    df["log_ret"] = np.log(df["close"]).diff()
    df["roll_vol"] = df["log_ret"].rolling(VOL_WINDOW).std()
    return df
 
 
# ---------------- forecast + signal (model-agnostic seam) ----------------
def get_forecast(df, horizon=FORECAST_HORIZON, lookback=TREND_LOOKBACK):
    recent = df.tail(lookback)
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent["close"].to_numpy(), 1)
    fx = np.arange(len(recent), len(recent) + horizon)
    last_date = df["obs_date"].iloc[-1]
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({
        "obs_date": [last_date, *dates],
        "forecast": [df["close"].iloc[-1], *(slope * fx + intercept)],
    })
 
 
def get_signal(df, lookback=TREND_LOOKBACK):
    if len(df) <= lookback:
        return "Not enough history to assess a trend."
    now, then = df["close"].iloc[-1], df["close"].iloc[-lookback]
    pct = (now - then) / then * 100
    trend = "trending upward" if pct > 2 else "trending downward" if pct < -2 else "roughly flat"
    return f"Over the last {lookback} trading days, arabica has moved {pct:+.1f}% — {trend}."
 
 
# ---------------- page ----------------
st.set_page_config(page_title="Coffee Price Dashboard", page_icon="☕", layout="wide")
st.title("☕ Coffee Price Dashboard")
st.caption(SYMBOL_LABEL)
 
df = load_prices(SYMBOL)
if df.empty:
    st.warning("No data found. Has the backfill been run?")
    st.stop()
 
tab1, tab2 = st.tabs(["📈 Price", "📊 Volatility & Analysis"])
 
# ===== Tab 1: Price =====
with tab1:
    latest, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    change = latest["close"] - prev["close"]
    pct = (change / prev["close"] * 100) if prev["close"] else 0.0
 
    c1, c2 = st.columns(2)
    c1.metric(f"Latest close ({latest['obs_date'].date()})",
              f"{latest['close']:.2f}", f"{change:+.2f} ({pct:+.2f}%)")
    c2.metric("History", f"{len(df)} trading days")
 
    st.info(get_signal(df))
 
    fc = get_forecast(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["obs_date"], y=df["close"], name="History",
                             line=dict(color="#6F4E37")))
    fig.add_trace(go.Scatter(x=fc["obs_date"], y=fc["forecast"],
                             name="Naive trend (indicative only)",
                             line=dict(color="#C8964F", dash="dash")))
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      yaxis_title="US cents / lb")
    if len(df) > 126:
        fig.update_xaxes(range=[df["obs_date"].iloc[-126], fc["obs_date"].iloc[-1]])
    st.plotly_chart(fig, use_container_width=True)
 
    st.caption(
        "⚠️ The dashed line is a simple trend baseline. Backtesting shows this "
        "kind of price extrapolation does **not** beat a naive forecast — see the "
        "Volatility & Analysis tab for what *is* predictable."
    )
 
# ===== Tab 2: Volatility & Analysis =====
with tab2:
    st.markdown(
        "### Price is hard to predict — but volatility isn't\n"
        "The price *level* is close to a random walk. But the *size* of daily "
        "moves (volatility) **clusters** — calm follows calm, turbulent follows "
        "turbulent — which makes it forecastable."
    )
 
    dv = df.dropna(subset=["roll_vol"])
    latest_vol = dv["roll_vol"].iloc[-1]
    avg_vol = dv["roll_vol"].tail(252).mean()
    ratio = latest_vol / avg_vol if avg_vol else 1.0
    regime = "elevated" if ratio > 1.1 else "calm" if ratio < 0.9 else "normal"
    ann = latest_vol * np.sqrt(252) * 100  # annualized, as a %
 
    st.metric(f"Annualized volatility ({VOL_WINDOW}-day, latest)",
              f"{ann:.1f}%",
              f"{(ratio - 1) * 100:+.0f}% vs 1-yr avg ({regime})")
 
    vfig = go.Figure()
    vfig.add_trace(go.Scatter(x=dv["obs_date"], y=dv["roll_vol"],
                              line=dict(color="#B5651D"), name="rolling vol"))
    vfig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                       yaxis_title="std of daily returns",
                       title=f"{VOL_WINDOW}-day rolling volatility (clustering is visible)")
    st.plotly_chart(vfig, use_container_width=True)
 
    st.markdown("#### Walk-forward backtest results")
    st.caption("Predicting price level — nothing beats the naive baseline:")
    st.table(PRICE_BACKTEST)
    st.caption("Predicting volatility — models beat the persistence baseline:")
    st.table(VOL_BACKTEST)
 
    st.markdown(
        "**Takeaway:** direction has no memory, but magnitude does. The right move "
        "wasn't a fancier price model — it was forecasting a target that's actually "
        "predictable. Full methodology in the project repo."
    )
 