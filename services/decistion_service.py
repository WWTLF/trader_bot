from models.position import Position
from datetime import date
from datetime import datetime
from repositories.position_repo import PositionRepository
import psycopg

class DecisionService:
    def __init__(self, ticker: str, conn: psycopg.Connection):
        self.ticker = ticker
        self.position = None
        self.profit = 0.0
        self.not_trust_th = 0.3
        self.not_trust_date = date(1971, 1, 1)
        self.conn = conn
        self.position_repo = PositionRepository(conn)


    def deserialize(self):
        last_position = self.position_repo.get_last_position(self.ticker)
        if last_position.opened:
            self.postion = last_position
        else:
            self.position = None

    def closeLastPosition(self, current_date: date, price: float):
        self.position.close_price = price
        self.position.closed_at = current_date
        if self.position.position_type == 'long':
            self.position.profit = self.position.qty * self.position.close_price - self.position.qty * self.position.open_price
        else: 
            self.position.profit = self.position.qty * self.position.open_price - self.position.qty * self.position.close_price 
        self.position.opened = False
        self.profit = self.profit + self.position.profit
        self.position = None
        self.position_repo.update(self.position)

    def decide(self, current_date: date,signal: int, price: float, qty: int, not_trust: float)-> tuple[bool, int]:

        if not_trust > self.not_trust_th:
            self.not_trust_date = current_date

        skip = False
        delta = current_date-datetime.combine(self.not_trust_date, datetime.min.time())
        if delta.days < 3:
            skip = True

        if signal == 1:
            if not self.position:
                if skip:
                    return False, 0
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
                self.position_repo.save(self.position)
                return False, 1
            else:
                if self.position.position_type == 'short':
                    self.closeLastPosition(current_date=current_date, price=price)
                    if skip:
                        return True, 0
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
                    self.position_repo.update(self.position)
                    return True, 1

            
        elif signal == -1:
            if not self.position:
                if skip:
                        return False, 0
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
                self.position_repo.update(self.position)
                return False, -1
            else:
                if self.position.position_type == 'long':
                    self.closeLastPosition(current_date=current_date, price=price)
                    if skip:
                        return True, 0
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
                    self.position_repo.save(self.position)
                    return True, -1
        return False, 0