# CREATE TABLE IF NOT EXISTS stock_data (
#     ticker varchar(10) not null,
#     stock_date datetime not null,
#     open FLOAT not null,
#     close FLOAT not null,
#     high FLOAT not null,
#     low FLOAT not null,
#     volume FLOAT not null
# );

from dataclasses import dataclass
from datetime import date

@dataclass
class StockData:
    ticker: str
    stock_date: date
    open: float
    close: float
    high: float
    low: float
    volume: float