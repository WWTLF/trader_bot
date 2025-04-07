import psycopg
from models.decision import Decision
from typing import Optional, List
import json

class DecistionRepository:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def save(self, dec: Decision) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO decision (ticker, position_id, decision_date, filtered_pred_price, 
                                       filtered_covariance, signal, close_prev)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                dec.ticker,
                dec.position_id,
                dec.decision_date,
                json.dumps(dec.filtered_pred_price),
                json.dumps(dec.filtered_covariance),
                dec.signal,
                dec.close_prev
            ))
            dec.id = cur.fetchone()[0]
        self.conn.commit()
        return dec.id

    def update(self, dec: Decision) -> None:
        if dec.id is None:
            raise ValueError("Cannot update Decistion without ID")

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE decision
                SET ticker = %s,
                    position_id = %s,
                    decision_date = %s,
                    filtered_pred_price = %s,
                    filtered_covariance = %s,
                    signal = %s,
                    close_prev = %s
                WHERE id = %s
            """, (
                dec.ticker,
                dec.position_id,
                dec.decision_date,
                json.dumps(dec.filtered_pred_price),
                json.dumps(dec.filtered_covariance),
                dec.signal,
                dec.close_prev,
                dec.id
            ))
        self.conn.commit()

    def get_by_id(self, id: int) -> Optional[Decision]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, ticker, position_id, decision_date, filtered_pred_price, filtered_covariance, signal, close_prev FROM decision WHERE id = %s", (id,))
            row = cur.fetchone()
            if row:
                return Decision(
                    id=row[0],
                    ticker=row[1],
                    position_id=row[2],
                    decision_date=row[3],
                    filtered_pred_price=json.loads(row[4]),
                    filtered_covariance=json.loads(row[5]),
                    signal=row[6],
                    close_prev=row[7]
                )
            return None

    def delete(self, id: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM decision WHERE id = %s", (id,))
        self.conn.commit()