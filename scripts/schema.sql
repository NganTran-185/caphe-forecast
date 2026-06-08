CREATE SCHEMA IF NOT EXISTS market_data;

CREATE TABLE IF NOT EXISTS market_data.prices (
    obs_date    DATE          NOT NULL,
    symbol      TEXT          NOT NULL,
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4) NOT NULL,
    volume      BIGINT,
    source      TEXT          NOT NULL,
    scraped_at  TIMESTAMPTZ   NOT NULL DEFAULT now(),
    CONSTRAINT pk_prices PRIMARY KEY (obs_date, symbol, source),
    CONSTRAINT chk_close_positive CHECK (close > 0)
);