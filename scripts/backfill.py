from __future__ import annotations
 
from fetch_futures import fetch_futures
from db import upsert_prices, count_rows
 
 
def main() -> None:
    print("Fetching from yfinance...")
    rows = fetch_futures()          # uses SYMBOLS + DEFAULT_START from fetch_futures.py
    print(f"Fetched {len(rows)} rows.")
 
    if not rows:
        print("Nothing fetched — check tickers / network. Aborting.")
        return
 
    before = count_rows()
    upsert_prices(rows)
    after = count_rows()
 
    print(f"Row count: {before} -> {after}")
    print("Backfill complete.")
 
 
if __name__ == "__main__":
    main()