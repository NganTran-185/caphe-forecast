from __future__ import annotations
 
import os
import warnings
 
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
 
warnings.filterwarnings("ignore")
 
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
 
load_dotenv()
SYMBOL = "KC=F"
HORIZON = 21          # forecast window (~1 trading month)
MIN_TRAIN = 250
EWMA_LAMBDA = 0.94
REGIME_WINDOW = 63    # window used to judge the prevailing regime
 
# Regime cut-points = 25th / 75th percentiles of the 63d rolling-vol series.
# NOTE: these are computed from the full sample, so they are used for LABELLING
# results after the fact, never as an input to any forecast. No look-ahead bias
# enters the predictions themselves.
CALM_MAX = 0.0207
TURBULENT_MIN = 0.0244
 
 
def load_closes(symbol: str) -> np.ndarray:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check your .env)")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT close FROM market_data.prices "
            "WHERE symbol = %s ORDER BY obs_date",
            (symbol,),
        )
        rows = cur.fetchall()
    return np.array([float(r[0]) for r in rows])
 
 
def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))
 
 
def realized_vol(returns: np.ndarray) -> float:
    return float(np.std(returns)) if len(returns) > 1 else 0.0
 
 
# ---------------- forecasters: history + horizon -> predicted vol ----------------
def persistence(hist: np.ndarray, horizon: int) -> float:
    """Baseline: next-window vol = vol of the last `horizon` returns."""
    return realized_vol(hist[-horizon:])
 
 
def historical_30d(hist: np.ndarray, horizon: int) -> float:
    """4th baseline: simple 30-day rolling historical volatility.
    A longer, equally-weighted window — the classic 'textbook' estimate."""
    return realized_vol(hist[-30:])
 
 
def ewma(hist: np.ndarray, horizon: int, lam: float = EWMA_LAMBDA) -> float:
    var = hist[0] ** 2
    for r in hist[1:]:
        var = lam * var + (1.0 - lam) * r ** 2
    return float(np.sqrt(var))
 
 
def garch(hist: np.ndarray, horizon: int) -> float:
    scaled = hist * 100.0
    try:
        res = arch_model(scaled, mean="Constant", vol="GARCH",
                         p=1, q=1, dist="normal").fit(disp="off")
        fc = res.forecast(horizon=horizon, reindex=False)
        return float(np.sqrt(np.mean(fc.variance.values[-1]))) / 100.0
    except Exception:
        return ewma(hist, horizon)
 
 
# ---------------- regime classification ----------------
def classify_regime(hist: np.ndarray) -> str:
    """Regime PREVAILING AT FORECAST TIME, judged only from data the model could
    see. Classifying by realised future vol would be look-ahead bias."""
    v = realized_vol(hist[-REGIME_WINDOW:])
    if v < CALM_MAX:
        return "calm"
    if v > TURBULENT_MIN:
        return "turbulent"
    return "normal"
 
 
def walk_forward(returns: np.ndarray, forecaster, horizon: int = HORIZON,
                 min_train: int = MIN_TRAIN, step: int = HORIZON) -> pd.DataFrame:
    """One row per window: prediction, actual, and the regime at forecast time."""
    rows = []
    cutoff = min_train
    while cutoff + horizon <= len(returns):
        hist = returns[:cutoff]
        rows.append({
            "pred": forecaster(hist, horizon),
            "actual": realized_vol(returns[cutoff:cutoff + horizon]),
            "regime": classify_regime(hist),
        })
        cutoff += step
    return pd.DataFrame(rows)
 
 
def rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((df["pred"] - df["actual"]) ** 2)))
 
 
def mae(df: pd.DataFrame) -> float:
    return float(np.mean(np.abs(df["pred"] - df["actual"])))
 
 
def main() -> None:
    rets = log_returns(load_closes(SYMBOL))
    print(f"{SYMBOL}: {len(rets)} daily returns\n")
 
    models = {
        "persistence (baseline)": persistence,
        "historical 30d": historical_30d,
        "EWMA (l=0.94)": ewma,
    }
    if HAS_ARCH:
        models["GARCH(1,1)"] = garch
    else:
        print("(arch not installed — skipping GARCH)\n")
 
    results = {name: walk_forward(rets, fn) for name, fn in models.items()}
 
    # window counts per regime (same for every model)
    counts = results["persistence (baseline)"]["regime"].value_counts()
    print("Forecast windows per regime:")
    for r in ["calm", "normal", "turbulent"]:
        print(f"  {r:<10} {counts.get(r, 0):>3}")
 
    # ---- aggregate ----
    print("\n" + "=" * 64)
    print("AGGREGATE (all regimes pooled)")
    print(f"{'model':<24}{'MAE':>10}{'RMSE':>10}{'vs base':>12}")
    print("-" * 64)
    base = rmse(results["persistence (baseline)"])
    for name, df in results.items():
        r = rmse(df)
        delta = "—" if name.startswith("persistence") else f"{(r/base - 1)*100:+.1f}%"
        print(f"{name:<24}{mae(df):>10.5f}{r:>10.5f}{delta:>12}")
 
    # ---- disaggregated scorecard ----
    print("\n" + "=" * 64)
    print("SCORECARD — RMSE by regime (vs. persistence within each regime)")
    header = f"{'model':<24}" + "".join(f"{r:>13}" for r in ["calm", "normal", "turbulent"])
    print(header)
    print("-" * 64)
 
    base_by_regime = {
        r: rmse(results["persistence (baseline)"].query("regime == @r"))
        for r in ["calm", "normal", "turbulent"]
        if (results["persistence (baseline)"]["regime"] == r).any()
    }
 
    for name, df in results.items():
        cells = ""
        for r in ["calm", "normal", "turbulent"]:
            sub = df[df["regime"] == r]
            if sub.empty:
                cells += f"{'n/a':>13}"
                continue
            val = rmse(sub)
            if name.startswith("persistence"):
                cells += f"{val:>13.5f}"
            else:
                delta = (val / base_by_regime[r] - 1) * 100
                cells += f"{delta:>+12.1f}%"
        print(f"{name:<24}{cells}")
 
    print(
        "\nPersistence row shows raw RMSE; other rows show % vs. persistence in"
        "\nTHAT regime (negative = better). Check whether any model's edge shrinks"
        "\nor reverses in the turbulent column — that is the validation question."
    )
 
 
if __name__ == "__main__":
    main()