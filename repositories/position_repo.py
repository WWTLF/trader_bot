from typing import Optional, List
from models.position import Position
import psycopg
# from db import get_conn

class PositionRepository:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def save(self, pos: Position) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO position (
                    ticker, opened_at, closed_at, position_type, 
                    open_price, close_price, qty, opened, profit
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, opened_at) DO UPDATE
                SET 
                    closed_at = EXCLUDED.closed_at,
                    position_type = EXCLUDED.position_type,
                    open_price = EXCLUDED.open_price,
                    close_price = EXCLUDED.close_price,
                    qty = EXCLUDED.qty,
                    opened = EXCLUDED.opened,
                    profit = EXCLUDED.profit
                RETURNING id;
            """, (
                pos.ticker,
                pos.opened_at,
                pos.closed_at,
                pos.position_type,
                pos.open_price,
                pos.close_price,
                pos.qty,
                pos.opened,
                pos.profit
            ))
            pos.id = cur.fetchone()[0]
        self.conn.commit()
        return pos.id

    def update(self, pos: Position) -> None:
        if pos.id is None:
            raise ValueError("Cannot update Position without ID")

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE position
                SET ticker = %s,
                    opened_at = %s,
                    closed_at = %s,
                    position_type = %s,
                    open_price = %s,
                    close_price = %s,
                    qty = %s,
                    opened = %s,
                    profit = %s
                WHERE id = %s
            """, (
                pos.ticker,
                pos.opened_at,
                pos.closed_at,
                pos.position_type,
                pos.open_price,
                pos.close_price,
                pos.qty,
                pos.opened,
                pos.profit,
                pos.id
            ))
        self.conn.commit()

    def get_by_id(self, id: int) -> Optional[Position]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, opened_at, closed_at, position_type,
                       open_price, close_price, qty, opened, profit
                FROM position
                WHERE id = %s
            """, (id,))
            row = cur.fetchone()
            if row:
                return Position(*row)
            return None

    def delete(self, id: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM position WHERE id = %s", (id,))
        self.conn.commit()



    def get_last(self, ticker: str) -> Optional[Position]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, opened_at, closed_at, position_type,
                       open_price, close_price, qty, opened, profit
                FROM position
                WHERE ticker = %s
                ORDER BY opened_at DESC
                LIMIT 1
            """, (ticker,))
            row = cur.fetchone()
            if row:
                return Position(*row)
            return None