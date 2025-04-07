from dataclasses import dataclass
from datetime import date
from typing import Optional

# Define your Position model using dataclass
@dataclass
class Decision:
    id: Optional[int]
    decision_date: date
    ticker: str
    position_id: Optional[int]
    filtered_pred_price: dict
    filtered_covariance: dict
    signal: int
    close_prev: bool