from dataclasses import dataclass
from datetime import date
from typing import Optional

# Define your Position model using dataclass
@dataclass
class Position:
    id: Optional[int]
    ticker: str
    opened_at: date
    closed_at: Optional[date]
    position_type: str
    open_price: float
    close_price: Optional[float]
    qty: int
    opened: bool
    profit: Optional[float]