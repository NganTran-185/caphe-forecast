from __future__ import annotations
 
import os
import warnings
 
import numpy as np
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
HORIZON = 21          
EWMA_LAMBDA = 0.94    # RiskMetrics standard
 
 
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
 
 
def persistence(returns_hist: np.ndarray, horizon: int) -> float:
    """Baseline: next vol = realized vol of the last `horizon` returns."""
    return realized_vol(returns_hist[-horizon:])
 
 
def ewma(returns_hist: np.ndarray, horizon: int, lam: float = EWMA_LAMBDA) -> float:
    """EWMA: variance_t = lam*variance_{t-1} + (1-lam)*return^2. Adapts fast."""
    var = returns_hist[0] ** 2
    for r in returns_hist[1:]:
        var = lam * var + (1.0 - lam) * r ** 2
    return float(np.sqrt(var))
 
 
def garch(returns_hist: np.ndarray, horizon: int) -> float:
    """GARCH(1,1): the classic model for volatility clustering.
    arch wants percentage-scaled returns (~1-3), so scale x100 to fit, then
    scale the prediction back to daily-return units to stay comparable."""
    scaled = returns_hist * 100.0
    try:
        res = arch_model(scaled, mean="Constant", vol="GARCH",
                         p=1, q=1, dist="normal").fit(disp="off")
        fc = res.forecast(horizon=horizon, reindex=False)
        daily_var_pct = fc.variance.values[-1]          
        mean_var_pct = float(np.mean(daily_var_pct))
        return float(np.sqrt(mean_var_pct)) / 100.0     
    except Exception:
        return ewma(returns_hist, horizon)             
 
 
def mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.array(pred) - np.array(actual))))
 
 
def rmse(pred, actual) -> float:
    return float(np.sqrt(np.mean((np.array(pred) - np.array(actual)) ** 2)))
 
 
def walk_forward_vol(returns: np.ndarray, forecaster, horizon: int = HORIZON,
                     min_train: int = 250, step: int = HORIZON):
    """Forecast sees returns BEFORE the cutoff; actual vol is from AFTER it."""
    preds, actuals = [], []
    cutoff = min_train
    while cutoff + horizon <= len(returns):
        history = returns[:cutoff]
        actual = realized_vol(returns[cutoff:cutoff + horizon])
        preds.append(forecaster(history, horizon))
        actuals.append(actual)
        cutoff += step
    return preds, actuals
 
 
def main() -> None:
    prices = load_closes(SYMBOL)
    rets = log_returns(prices)
    print(f"Loaded {len(prices)} prices -> {len(rets)} daily returns for {SYMBOL}\n")
 
    models = {"persistence (baseline)": persistence, "EWMA (lambda=0.94)": ewma}
    if HAS_ARCH:
        models["GARCH(1,1)"] = garch
    else:
        print("(arch not installed — skipping GARCH. `pip install arch` to include it.)\n")
 
    print(f"{'model':<24}{'MAE':>11}{'RMSE':>11}   vs baseline")
    print("-" * 62)
    base = None
    for name, fn in models.items():
        p, a = walk_forward_vol(rets, fn)
        m, r = mae(p, a), rmse(p, a)
        if base is None:
            base = r
            verdict = "(baseline)"
        else:
            diff = (r - base) / base * 100
            verdict = f"{diff:+.1f}% RMSE {'worse' if diff > 0 else 'better'}"
        print(f"{name:<24}{m:>11.5f}{r:>11.5f}   {verdict}")
 
    print(
        "\nDaily-return scale (~0.01-0.03). Volatility is forecastable because it"
        "\nclusters — calm follows calm, turbulent follows turbulent."
    )
 
 
if __name__ == "__main__":
    main()
 