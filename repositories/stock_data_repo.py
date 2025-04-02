
import pandas as pd
from datetime import date

from models.stock_data import StockData
from utils.outliers import get_rid_of_outliers

class StockDataRepo:
    def __init__(self, conn):
        self.conn = conn

    def Upsert(self, stock_data: StockData):
       
        result = self.conn.execute("""
                INSERT INTO stock_data(ticker, stock_date, open, close, high, low, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, stock_date) DO NOTHING
            """, (
                stock_data.ticker,
                stock_data.stock_date,
                stock_data.open,
                stock_data.close,
                stock_data.high,
                stock_data.low,
                stock_data.volume
            ), prepare=True)

        # row = result.fetchone()
        self.conn.commit()
    

    def get_all_stock(self) ->pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute("select ticker, stock_date, open, close, high, low, volume from stock_data;")
        rows = cur.fetchall()
        cur.close()
        self.conn.close()
        stock_df = pd.DataFrame(rows, columns=["ticker", "stock_date", "open", "close", "high", "low", "volume"])
        stock_df['stock_date'] = pd.to_datetime(stock_df['stock_date'])
        # Сглаживаем выбросы
        get_rid_of_outliers(stock_df)
        stock_df.set_index(['ticker', 'stock_date'], inplace=True)
        # Filtering by stock_date range (across all tickers)
        stock_df = stock_df.sort_index()
        return stock_df
    
    def get_last_ticker(self, ticker: str) -> StockData:
        cur = self.conn.cursor()
        cur.execute("select ticker, stock_date, open, close, high, low, volume from stock_data where ticker = %s order by stock_date desc limit 1", (ticker,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return StockData(row[0], row[1], row[2], row[3], row[4], row[5], row[6])

