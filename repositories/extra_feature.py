from dataclasses import dataclass
from typing import Optional, List
from datetime import date
from models.extra_features import ExtraFeature
import psycopg
import pandas as pd


class ExtraFeatureRepository:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def save(self, feature: ExtraFeature) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO extra_feature (
                    ticker, stock_date, feature_name, feature_value
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, stock_date, feature_name)
                DO UPDATE SET feature_value = EXCLUDED.feature_value
                RETURNING id
            """, (
                feature.ticker,
                feature.stock_date,
                feature.feature_name,
                feature.feature_value
            ))
            feature.id = cur.fetchone()[0]
        self.conn.commit()
        return feature.id

    def update(self, feature: ExtraFeature) -> None:
        if feature.id is None:
            raise ValueError("Cannot update ExtraFeature without ID")

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE extra_feature
                SET ticker = %s,
                    stock_date = %s,
                    feature_name = %s,
                    feature_value = %s
                WHERE id = %s
            """, (
                feature.ticker,
                feature.stock_date,
                feature.feature_name,
                feature.feature_value,
                feature.id
            ))
        self.conn.commit()

    def get_by_id(self, id: int) -> Optional[ExtraFeature]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, stock_date, feature_name, feature_value
                FROM extra_feature
                WHERE id = %s
            """, (id,))
            row = cur.fetchone()
            if row:
                return ExtraFeature(*row)
            return None

    def delete(self, id: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM extra_feature WHERE id = %s", (id,))
        self.conn.commit()

    def get_all_for_ticker_and_date(self, ticker: str, stock_date_from: date, stock_date_to: date) -> pd.DataFrame:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT stock_date, feature_name, feature_value
                FROM extra_feature
                WHERE ticker = %s AND stock_date >= %s and stock_date <= %s
            """, (ticker, stock_date_from, stock_date_to))
            rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=['stock_date', 'feature_name', 'feature_value'])
        if df.empty:
            return pd.DataFrame()  # Вернём пустой DataFrame, если данных нет
        
        df['stock_date'] = pd.to_datetime(df['stock_date'])
        # df.set_index(['stock_date'], inplace=True)

        # Pivot: feature_name → колонки, stock_date → индекс
        df_pivot = df.pivot(index='stock_date', columns='feature_name', values='feature_value')

        # df_pivot.reset_index(inplace=True)  # если хочешь вернуть stock_date как колонку
        return df_pivot
    
    def bulk_save(self, df: pd.DataFrame, feature_names: List[str]) -> None:
        values = []
        for _, row in df.iterrows():
            # stock_date может быть в индексе или колонке
            stock_date = row['stock_date'] if 'stock_date' in row else row.name
            ticker = row['ticker']
            for feature in feature_names:
                feature_value = row[feature]
                values.append((
                    ticker,
                    stock_date,
                    feature,
                    float(feature_value)
                ))

        with self.conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO extra_feature (ticker, stock_date, feature_name, feature_value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, stock_date, feature_name)
                DO UPDATE SET feature_value = EXCLUDED.feature_value
            """, values)

        self.conn.commit()


    def get_one_by_ticker_and_date(self, ticker: str, stock_date: date, feature_name: str) -> Optional[ExtraFeature]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, stock_date, feature_name, feature_value
                FROM extra_feature
                WHERE ticker = %s AND stock_date = %s AND feature_name = %s
            """, (ticker, stock_date, feature_name))
            row = cur.fetchone()

        if row:
            return ExtraFeature(*row)
        return None