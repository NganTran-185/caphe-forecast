
from __future__ import annotations
 
import os
 
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
 
load_dotenv()
SYMBOL = "KC=F"
WINDOW = 63   
 
 
def load_df(symbol: str) -> pd.DataFrame:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check your .env)")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
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
 
 
def main() -> None:
    df = load_df(SYMBOL)
    # returns are 1 shorter than prices; align by dropping the first row
    df["log_ret"] = np.log(df["close"]).diff()
    df["roll_vol"] = df["log_ret"].rolling(WINDOW).std()
    dv = df.dropna(subset=["roll_vol"]).reset_index(drop=True)
 
    print(f"{SYMBOL}: {len(df)} prices, {df['obs_date'].min().date()} -> "
          f"{df['obs_date'].max().date()}")
    print(f"Rolling-vol series ({WINDOW}d): {len(dv)} points\n")
 
    # --- distribution: gives data-driven regime thresholds ---
    q = dv["roll_vol"].quantile([0.25, 0.50, 0.75, 0.90]).round(5)
    print("Rolling-vol distribution:")
    for k, v in q.items():
        print(f"  {int(k*100):>3}th percentile : {v:.5f}")
    print()
 
    # --- the top volatility episodes, with dates ---
    print("10 highest-volatility days (peak of each episode may repeat):")
    top = dv.nlargest(10, "roll_vol")[["obs_date", "roll_vol", "close"]]
    for _, r in top.iterrows():
        print(f"  {r['obs_date'].date()}   vol={r['roll_vol']:.5f}   close={r['close']:.2f}")
    print()
 
    # --- the early spike seen around plot index ~140-200 ---
    print("Window around the early spike (plot index 120-210):")
    seg = dv.iloc[120:210]
    peak = seg.loc[seg["roll_vol"].idxmax()]
    print(f"  segment spans {seg['obs_date'].iloc[0].date()} -> {seg['obs_date'].iloc[-1].date()}")
    print(f"  peak on {peak['obs_date'].date()} at vol={peak['roll_vol']:.5f}\n")
 
    # --- the spike at the end of the sample ---
    print("Last 10 observations (the end-of-sample spike):")
    for _, r in dv.tail(10).iterrows():
        print(f"  {r['obs_date'].date()}   vol={r['roll_vol']:.5f}   close={r['close']:.2f}")
 
    latest = dv["roll_vol"].iloc[-1]
    pct = (dv["roll_vol"] < latest).mean() * 100
    print(f"\nCurrent rolling vol = {latest:.5f} "
          f"({pct:.1f}th percentile of the sample)")
    print(f"Sample median = {dv['roll_vol'].median():.5f}")
 
 
if __name__ == "__main__":
    main()