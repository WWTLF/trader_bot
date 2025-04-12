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
    profit FLOAT8,
    UNIQUE (ticker, opened_at)
);

CREATE TABLE decision (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    decision_date DATE NOT NULL,
    position_id INT,
    filtered_pred_price JSONB NOT NULL,
    filtered_covariance JSONB NOT NULL,
    signal INT NOT NULL,
    close_prev BOOLEAN NOT NULL
);

CREATE TABLE extra_feature (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    stock_date DATE NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value float NOT NULL
);

-- Уникальность фичи на дату и тикер
CREATE UNIQUE INDEX uniq_extra_feature
    ON extra_feature (ticker, stock_date, feature_name);