from __future__ import annotations
 
import os
 
import psycopg
from dotenv import load_dotenv
 
load_dotenv()  # reads .env
 
 
def _connection_string() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Put it in your .env file "
            "(the same Neon connection string psql uses)."
        )
    return url
 

UPSERT_SQL = """
    INSERT INTO market_data.prices
        (obs_date, symbol, open, high, low, close, volume, source)
    VALUES
        (%(obs_date)s, %(symbol)s, %(open)s, %(high)s, %(low)s,
         %(close)s, %(volume)s, %(source)s)
    ON CONFLICT (obs_date, symbol, source) DO UPDATE SET
        open   = EXCLUDED.open,
        high   = EXCLUDED.high,
        low    = EXCLUDED.low,
        close  = EXCLUDED.close,
        volume = EXCLUDED.volume;
"""
 
def upsert_prices(rows: list[dict]) -> int:
    """
    Upsert a list of row-dicts into market_data.prices.
    Returns the number of rows sent. Safe to run repeatedly.
    """
    if not rows:
        print("No rows to upsert.")
        return 0
 
    with psycopg.connect(_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
        conn.commit()
 
    print(f"Upserted {len(rows)} rows.")
    return len(rows)
 
 
def count_rows() -> int:
    """Quick check: how many rows are currently in the table?"""
    with psycopg.connect(_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM market_data.prices;")
            (n,) = cur.fetchone()
    return n
 
 
if __name__ == "__main__":
    print(f"Connected. Rows in market_data.prices: {count_rows()}")
 