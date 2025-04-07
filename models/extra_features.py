from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class ExtraFeature:
    id: Optional[int]
    ticker: str
    stock_date: date
    feature_name: str
    feature_value: float