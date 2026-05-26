"""Smoke test: can Python reach Postgres?"""
from sqlalchemy import text
from caphe_forecast.utils.db import get_engine


def main():
    engine = get_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
        timescale = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname='timescaledb'")
        ).scalar()
    print("✅ Connected to Postgres")
    print(f"   Postgres version: {version[:70]}...")
    print(f"   TimescaleDB extension: {timescale or 'NOT INSTALLED'}")


if __name__ == "__main__":
    main()