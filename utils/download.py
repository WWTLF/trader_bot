from datetime import date
import yfinance as yf
from dateutil.relativedelta import relativedelta
from utils.outliers import get_rid_of_outliers
import pandas as pd
from repositories.stock_data_repo import StockDataRepo
from models.stock_data import StockData
from db import get_conn

def download_all():
    tickers_to_download = ['AAPL', 'GOOG','AMZN', 'MSFT', 'AMD', 'NVDA', 'IBM']
    conn = get_conn()
    for t in tickers_to_download:
        preload_date_for_ticker(conn, t,  date.today())
    conn.close()

# Скачивание данных о котировках
def download_stock_data(ticker, start_date: date, end_date: date):
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d", threads=True, auto_adjust=True, group_by='Ticker')
    return data


def preload_date_for_ticker(conn, ticker: str, today: date) -> pd.DataFrame:
    print("downloading ", ticker)
    start_date = today

    stock_data_repo = StockDataRepo(conn)
    last_item = stock_data_repo.get_last_ticker(ticker)
    start_date = today - relativedelta(years=5) 
    if last_item is not None:
        start_date = last_item.stock_date
    print("last_date ", start_date)


    data = download_stock_data(ticker, start_date, today)[ticker]
    # Заполняем пропуски
    data.ffill(inplace=True)

    for index, row in data.iterrows():      
        stock_data_repo.Upsert(StockData(ticker, index, row['Open'], row['Close'], row['High'], row['Low'], row['Volume']))  


def get_all_stock() -> pd.DataFrame:
    conn = get_conn()
    stock_data_repo = StockDataRepo(conn)
    df =  stock_data_repo.get_all_stock()
    conn.close()
    return df