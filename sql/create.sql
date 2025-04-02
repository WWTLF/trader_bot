CREATE TABLE ml_model_config (
    id SERIAL PRIMARY KEY,
    model_name TEXT UNIQUE NOT NULL,
    config JSONB NOT NULL,
    path TEXT NOT NULL
);

CREATE TABLE stock_data (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    stock_date DATE NOT NULL,
    open NUMERIC(12, 4),
    close NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    volume BIGINT,
    UNIQUE (ticker, stock_date)
);


CREATE TABLE Position (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    opened_at DATE NOT NULL,
    closed_at DATE,
    position_type VARCHAR(20) NOT NULL,
    open_price FLOAT8 NOT NULL,
    close_price FLOAT8,
    qty INTEGER NOT NULL,
    opened BOOLEAN NOT NULL,
    profit FLOAT8
);