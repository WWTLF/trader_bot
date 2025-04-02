from models.position import Position
from datetime import date

class DecisionService:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.position = None
        self.profit = 0.0

    def closeLastPosition(self, current_date: date, price: float):
        self.position.close_price = price
        self.position.closed_at = current_date
        if self.position.position_type == 'long':
            self.position.profit = self.position.qty * self.position.close_price - self.position.qty * self.position.open_price
        else: 
            self.position.profit = self.position.qty * self.position.open_price - self.position.qty * self.position.close_price 
        self.position.opened = False
        self.profit = self.profit + self.position.profit
        # TODO: Persist

    def decide(self, current_date: date,signal: int, price: float, qty: int)-> tuple[bool, int]:
        if signal == 1:
            if not self.position:
                self.position = Position(
                    id = None,
                    ticker=self.ticker,
                    opened_at=current_date,
                    closed_at=None,
                    position_type='long',
                    open_price=price,
                    close_price=None,
                    qty=qty, 
                    opened=True,
                    profit=None)
                return False, 1
            else:
                if self.position.position_type == 'short':
                    self.closeLastPosition(current_date=current_date, price=price)
                    self.position = Position(id = None,
                        ticker=self.ticker,
                        opened_at=current_date,
                        closed_at=None,
                        position_type='long',
                        open_price=price,
                        close_price=None,
                        qty=qty, 
                        opened=True,
                        profit=None)
                    return True, 1
            
        elif signal == -1:
            if not self.position:
                self.position = Position(id = None,
                    ticker=self.ticker,
                    opened_at=current_date,
                    closed_at=None,
                    position_type='short',
                    open_price=price,
                    close_price=None,
                    qty=qty, 
                    opened=True,
                    profit=None)
                return False, -1
            else:
                if self.position.position_type == 'long':
                    self.closeLastPosition(current_date=current_date, price=price)
                    self.position = Position(id = None,
                        ticker=self.ticker,
                        opened_at=current_date,
                        closed_at=None,
                        position_type='short',
                        open_price=price,
                        close_price=None,
                        qty=qty, 
                        opened=True,
                        profit=None)
                    return True, -1
        return False, 0