from __future__ import annotations
 
import os
import warnings
 
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from arch import arch_model
 
warnings.filterwarnings("ignore")
load_dotenv()
 
SYMBOL = "KC=F"
Z_THRESHOLD = 3.0
METHOD = "in_sample"   # swap to 'walk_forward' when the leak-free version is built
 
 
def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check your .env)")
    return url
 
 
def load_prices(symbol: str) -> pd.DataFrame:
    with psycopg.connect(_db_url()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT obs_date, close FROM market_data.prices "
            "WHERE symbol = %s ORDER BY obs_date",
            (symbol,),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["obs_date", "close"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["close"] = df["close"].astype(float)
    return df
 
 
def garch_conditional_vol(returns: np.ndarray) -> np.ndarray:
    """In-sample daily conditional volatility from one GARCH(1,1) fit.
    SEAM: for the live detector, replace with a walk-forward version that fits
    on data up to day t-1 and forecasts day t's sigma (leak-free)."""
    scaled = returns * 100.0
    res = arch_model(scaled, mean="Constant", vol="GARCH",
                     p=1, q=1, dist="normal").fit(disp="off")
    return res.conditional_volatility / 100.0
 
 
def detect(df: pd.DataFrame, z_threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    out["log_ret"] = np.log(out["close"]).diff()
    rets = out["log_ret"].to_numpy()[1:]
    sigma = garch_conditional_vol(rets)
    out["garch_sigma"] = np.concatenate([[np.nan], sigma])
    out["z"] = (out["log_ret"] - np.nanmean(out["log_ret"])) / out["garch_sigma"]
    out["flagged"] = out["z"].abs() > z_threshold
    return out
 
 
PERSIST_SQL = """
INSERT INTO market_data.anomaly_scores
    (obs_date, symbol, log_ret, garch_sigma, z, direction, flagged, method)
VALUES
    (%(obs_date)s, %(symbol)s, %(log_ret)s, %(garch_sigma)s, %(z)s,
     %(direction)s, %(flagged)s, %(method)s)
ON CONFLICT (obs_date, symbol, method) DO UPDATE SET
    log_ret     = EXCLUDED.log_ret,
    garch_sigma = EXCLUDED.garch_sigma,
    z           = EXCLUDED.z,
    direction   = EXCLUDED.direction,
    flagged     = EXCLUDED.flagged,
    detected_at = now();
"""
 
 
def persist(result: pd.DataFrame, symbol: str = SYMBOL, method: str = METHOD) -> int:
    """Idempotent upsert of every evaluated day into anomaly_scores."""
    rows = []
    for _, r in result.dropna(subset=["z", "garch_sigma"]).iterrows():
        rows.append({
            "obs_date": r["obs_date"].date(),
            "symbol": symbol,
            "log_ret": float(r["log_ret"]),
            "garch_sigma": float(r["garch_sigma"]),
            "z": float(r["z"]),
            "direction": "up" if r["log_ret"] >= 0 else "down",
            "flagged": bool(r["flagged"]),
            "method": method,
        })
    if not rows:
        return 0
    with psycopg.connect(_db_url()) as conn:
        with conn.cursor() as cur:
            cur.executemany(PERSIST_SQL, rows)
        conn.commit()
    return len(rows)
 
 
def main() -> None:
    df = load_prices(SYMBOL)
    result = detect(df)
 
    n_flag = int(result["flagged"].sum())
    n_days = int(result["z"].notna().sum())
    print(f"{SYMBOL}: {n_days} days evaluated, {n_flag} flagged "
          f"at |z| > {Z_THRESHOLD} ({n_flag / n_days * 100:.1f}%)")
 
    flags = result[result["flagged"]].sort_values("z", key=lambda s: s.abs(), ascending=False)
    print("\nFlagged days (most extreme first):")
    for _, r in flags.iterrows():
        d = "up" if r["log_ret"] >= 0 else "down"
        print(f"  {r['obs_date'].date()}  z={r['z']:+.2f}  "
              f"return={r['log_ret']*100:+.2f}%  sigma={r['garch_sigma']*100:.2f}%  {d}")
 
    written = persist(result)
    print(f"\nUpserted {written} rows into market_data.anomaly_scores "
          f"(method='{METHOD}').")
 
 
if __name__ == "__main__":
    main()
 