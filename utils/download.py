import duckdb
from datetime import date
import yfinance as yf
from dateutil.relativedelta import relativedelta
from utils.outliers import get_rid_of_outliers
import pandas as pd


def download_all():
    db_connection = establish_db_connection()
    tickers_to_download = ['AAPL', 'GOOG','AMZN', 'MSFT', 'AMD', 'NVDA', 'IBM']
    for t in tickers_to_download:
        preload_date_for_ticker(db_connection, t,  date.today())
    db_connection.close()

# Скачивание данных о котировках
def download_stock_data(ticker, start_date: date, end_date: date):
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d", threads=True, auto_adjust=True, group_by='Ticker')
    return data


def preload_date_for_ticker(db_connection, ticker: str, today: date) -> pd.DataFrame:
    start_date = today
    last_item = db_connection.sql("select stock_date from stock_data where ticker like '{}' order by stock_date desc limit 1".format(ticker))   
    if last_item.df().shape[0] == 0:
        start_date = today - relativedelta(years=5)
    else:
        start_date = last_item.df().iloc[0]['stock_date']
    
    print(start_date, "->", ticker)
    
    data = download_stock_data(ticker, start_date, today)[ticker]
    # Заполняем пропуски
    data.ffill(inplace=True)

    for index, row in data.iterrows():        
        insert_statement = """
            insert into stock_data(ticker, stock_date, open, close, high, low, volume) values('{}', '{}', {}, {}, {}, {}, {})  ON CONFLICT DO NOTHING;
        """.format(
                   ticker,
                   index,
                   row['Open'],
                   row['Close'],
                   row['High'],
                   row['Low'],
                   row['Volume'])
        # print(insert_statement)
        db_connection.sql(insert_statement)

def establish_db_connection():
    con = duckdb.connect("./stock_data.db")

    create_table_statement = """
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker varchar(10) not null,
            stock_date datetime not null,
            open FLOAT not null,
            close FLOAT not null,
            high FLOAT not null,
            low FLOAT not null,
            volume FLOAT not null
        );
        
    """

    create_index_statement = """
        CREATE UNIQUE INDEX ticker_ts ON stock_data (ticker, stock_date);
    """

    con.sql(create_table_statement)

    try:
        con.sql(create_index_statement)
    except:
        print("index alredy exists")

    return con


def get_stock_by_range(start: date, end: date) -> pd.DataFrame:
    db_connection = establish_db_connection()
    sql = "select * from stock_data where stock_date >= '{}' and stock_date <= '{}'".format(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(sql)
    df = db_connection.sql(sql).df()
    # Сглаживаем выбросы
    get_rid_of_outliers(df)
    db_connection.close()
    return df