"""Database connection helper. Reads from .env via python-dotenv."""

import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()


def get_engine() -> Engine:
    user = os.environ["POSTGRES_USER"]
    pw = os.environ["POSTGRES_PASSWORD"]
    host = os.environ["POSTGRES_HOST"]
    port = os.environ["POSTGRES_PORT"]
    db = os.environ["POSTGRES_DB"]
    url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)