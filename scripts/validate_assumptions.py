from __future__ import annotations
 
import os
 
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
 
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
 
load_dotenv()
SYMBOL = "KC=F"
ALPHA = 0.05
PLOT_PATH = "docs/assumption_checks.png"
 
 
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
 
 
# Test 1: ADF — stationarity. Null hypothesis: series has a unit root
# (non-stationary). We want to REJECT it (p < 0.05) for returns.

def adf_test(series: np.ndarray, name: str) -> float:
    stat, p, *_ = adfuller(series)
    stationary = p < ALPHA
    print(f"\nADF test on {name}:")
    print(f"  statistic = {stat:.3f}   p-value = {p:.4g}")
    print(f"  -> {'STATIONARY (reject unit root)' if stationary else 'non-stationary (cannot reject)'}")
    return p
 
 

# Test 2: ARCH effects — volatility clustering. Null hypothesis: NO ARCH
# effects. We want to REJECT it (p < 0.05) -> clustering exists -> GARCH warranted.

def arch_tests(returns: np.ndarray) -> tuple[float, float]:
    resid = returns - returns.mean()
 
    lm_stat, lm_p, _f, _fp = het_arch(resid, nlags=10)
    print("\nARCH-effects — Engle's LM test (H0: no ARCH effects):")
    print(f"  LM statistic = {lm_stat:.2f}   p-value = {lm_p:.4g}")
    print(f"  -> {'ARCH EFFECTS PRESENT -> GARCH justified' if lm_p < ALPHA else 'no ARCH effects detected'}")
 
    lb = acorr_ljungbox(resid ** 2, lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    print("\nARCH-effects — Ljung-Box on squared returns (H0: no autocorrelation):")
    print(f"  p-value = {lb_p:.4g}")
    print(f"  -> {'squared returns autocorrelated -> clustering confirmed' if lb_p < ALPHA else 'no clustering detected'}")
 
    return lm_p, lb_p
 
# Test 3: structural breaks — visual. Rolling mean should sit near zero;
# rolling volatility shifting between levels reveals regime changes.

def structural_break_plot(returns: np.ndarray, window: int = 63) -> None:
    s = pd.Series(returns)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
 
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(roll_mean)
    ax[0].axhline(0, color="grey", lw=0.7)
    ax[0].set_title(f"Rolling mean of returns ({window}d) — should hover near 0")
    ax[1].plot(roll_std, color="firebrick")
    ax[1].set_title(f"Rolling volatility ({window}d) — regime shifts appear as level changes")
    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved structural-break plot -> {PLOT_PATH}")
 
 
def main() -> None:
    prices = load_closes(SYMBOL)
    log_ret = np.diff(np.log(prices))
    print(f"Loaded {len(prices)} prices -> {len(log_ret)} log returns for {SYMBOL}")
    print("=" * 66)
 
    # ADF: prices should be non-stationary, returns stationary
    adf_test(prices, "PRICE level (expect non-stationary)")
    p_ret = adf_test(log_ret, "log RETURNS (expect stationary)")
 
    # ARCH effects on returns
    lm_p, lb_p = arch_tests(log_ret)
 
    # structural-break visual
    structural_break_plot(log_ret)
 
    print("\n" + "=" * 66)
    print("ASSUMPTIONS TESTED — summary:")
    print(f"  Stationarity (returns): {'HOLDS' if p_ret < ALPHA else 'FRAGILE'} (ADF p={p_ret:.3g})")
    arch_ok = lm_p < ALPHA and lb_p < ALPHA
    print(f"  ARCH effects / clustering: {'PRESENT -> GARCH justified' if arch_ok else 'NOT clearly present'}")
    print("  Caveat: ADF assesses the whole sample; stationarity can be locally")
    print("  fragile around structural breaks (supply shocks) — see the saved plot.")
 
 
if __name__ == "__main__":
    main()
 