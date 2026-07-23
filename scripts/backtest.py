from __future__ import annotations
 
import os
 
import numpy as np
import psycopg
from dotenv import load_dotenv
 
load_dotenv()
SYMBOL = "KC=F"
 
 
def load_closes(symbol: str) -> np.ndarray:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check your .env)")
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT close FROM market_data.prices "
                "WHERE symbol = %s ORDER BY obs_date",
                (symbol,),
            )
            rows = cur.fetchall()
    return np.array([float(r[0]) for r in rows])
 
# Ba baseline can phai beat boi bat ke model nao

def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1], horizon) #Random: today = tmr
 
 
def drift(history: np.ndarray, horizon: int, lookback: int = 30) -> np.ndarray: #tmr = tday + avg change 
    recent = history[-lookback:]
    step = float(np.mean(np.diff(recent))) if len(recent) > 1 else 0.0
    return history[-1] + step * np.arange(1, horizon + 1)
 
 
def trend(history: np.ndarray, horizon: int, lookback: int = 30) -> np.ndarray: #tmr = tday + trend (displayed in dashboard)
    recent = history[-lookback:]
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    future_x = np.arange(len(recent), len(recent) + horizon)
    return slope * future_x + intercept
 
 
def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))
 
 
def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))
 
 

def walk_forward(series: np.ndarray, forecaster, horizon: int = 14,
                 min_train: int = 250, step: int = 14):
    preds, actuals = [], []
    cutoff = min_train
    while cutoff + horizon <= len(series):
        history = series[:cutoff]
        actual = series[cutoff:cutoff + horizon]
        preds.append(forecaster(history, horizon))
        actuals.append(actual)
        cutoff += step
    return np.concatenate(preds), np.concatenate(actuals)
 
 
def main() -> None:
    series = load_closes(SYMBOL)
    print(f"Loaded {len(series)} closing prices for {SYMBOL}\n")
 
    forecasters = {"naive (baseline)": naive, "drift": drift, "trend": trend}
 
    print(f"{'model':<18}{'MAE':>9}{'RMSE':>9}   vs baseline")
    print("-" * 50)
    base_rmse = None
    for name, fn in forecasters.items():
        preds, actuals = walk_forward(series, fn)
        m, r = mae(preds, actuals), rmse(preds, actuals)
        if base_rmse is None:
            base_rmse = r
            verdict = "(baseline)"
        else:
            diff = (r - base_rmse) / base_rmse * 100
            verdict = f"{diff:+.1f}% RMSE {'worse' if diff > 0 else 'better'}"
        print(f"{name:<18}{m:>9.2f}{r:>9.2f}   {verdict}")
 
    print(
        "\nLower is better. If a model can't beat the naive baseline, that's a"
        "\nreal, honest finding — daily futures prices are close to a random"
        "\nwalk in the short term."
    )
 
 
if __name__ == "__main__":
    main()