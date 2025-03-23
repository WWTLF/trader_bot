import psycopg
from psycopg_pool import ConnectionPool
import sys
import psycopg_pool
from psycopg_pool import PoolTimeout
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

DATABASE_URL = "postgresql://postgres:JewPs37kke@localhost:5432/trader_bot"

pool = ConnectionPool(DATABASE_URL, open=True, max_size=10)

def get_conn():
    return pool.getconn(timeout=120)



