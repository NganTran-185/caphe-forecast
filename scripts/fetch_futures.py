from __future__ import annotations
 
import yfinance as yf
 
 
# Yahoo Finance tickers.
#   KC=F -> Arabica (ICE Coffee C)
SYMBOLS = [
    "KC=F"       
]
 
SOURCE = "yfinance"
DEFAULT_START = "2021-01-01"   # ~3+ years of history for the backfill
 
 
def _num(v) -> float | None:
    """Coerce to float; turn NaN into None (NaN != NaN is the NaN test)."""
    return float(v) if v == v else None
 
 
def fetch_futures(symbols: list[str] = SYMBOLS, start: str = DEFAULT_START) -> list[dict]:
    """Pull daily OHLCV for each symbol and flatten into row-dicts."""
    rows: list[dict] = []
 
    for symbol in symbols:
        df = yf.Ticker(symbol).history(start=start, auto_adjust=False)
 
        if df.empty:
            print(f"WARNING: no data returned for {symbol!r} - skipping")
            continue
 
        for idx, r in df.iterrows():
            close = _num(r["Close"])
            if close is None or close <= 0:
                continue  
            vol = r.get("Volume")
            rows.append(
                {
                    "obs_date": idx.date(),          
                    "symbol": symbol,
                    "open": _num(r["Open"]),
                    "high": _num(r["High"]),
                    "low": _num(r["Low"]),
                    "close": close,
                    "volume": int(vol) if (vol is not None and vol == vol) else None,
                    "source": SOURCE,
                }
            )
 
    return rows
 
 
if __name__ == "__main__":
    data = fetch_futures()
 
    for symbol in sorted({row["symbol"] for row in data}):
        sym_rows = sorted(
            (r for r in data if r["symbol"] == symbol),
            key=lambda r: r["obs_date"],
        )
        print(
            f"\n{symbol}: {len(sym_rows)} rows, "
            f"{sym_rows[0]['obs_date']} -> {sym_rows[-1]['obs_date']}"
        )
        for r in sym_rows[-3:]:
            print(f"  {r['obs_date']}  close={r['close']:.2f}  vol={r['volume']}")
 
    print(f"\nTotal: {len(data)} rows")
    assert len(data) > 0, "no data fetched - check ticker symbols / network"
    print("checks passed")