from typing import Optional, List
from models.position import Position
from typing import Optional, List
from db import get_conn

class PositionRepo:
    @staticmethod
    def insert(position: Position) -> int:
        query = """
            INSERT INTO Position (
                ticker, opened_at, closed_at, position_type,
                open_price, close_price, qty, opened, profit
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        values = (
            position.ticker,
            position.opened_at,
            position.closed_at,
            position.position_type,
            position.open_price,
            position.close_price,
            position.qty,
            position.opened,
            position.profit
        )
        conn = get_conn
        with conn.cursor() as cur:
            cur.execute(query, values)
            new_id = cur.fetchone()[0]
            return new_id

    @staticmethod
    def get_by_id(position_id: int) -> Optional[Position]:
        query = "SELECT * FROM Position WHERE id = %s;"
        conn = get_conn
        with conn.cursor() as cur:
            cur.execute(query, (position_id,))
            row = cur.fetchone()
            if row:
                return Position(*row)
            return None

    @staticmethod
    def list_all() -> List[Position]:
        query = "SELECT * FROM Position ORDER BY opened_at DESC;"
        conn = get_conn
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return [Position(*row) for row in rows]
        

    def update(position: Position) -> bool:
        if position.id is None:
            raise ValueError("Cannot update position without an ID.")

        query = """
            UPDATE Position SET
                ticker = %s,
                opened_at = %s,
                closed_at = %s,
                position_type = %s,
                open_price = %s,
                close_price = %s,
                qty = %s,
                opened = %s,
                profit = %s
            WHERE id = %s;
        """
        values = (
            position.ticker,
            position.opened_at,
            position.closed_at,
            position.position_type,
            position.open_price,
            position.close_price,
            position.qty,
            position.opened,
            position.profit,
            position.id
        )
        conn = get_conn
        with conn.cursor() as cur:
            cur.execute(query, values)
            return cur.rowcount > 0  # Returns True if a row was updated