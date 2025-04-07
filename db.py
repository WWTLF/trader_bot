# import psycopg
from psycopg_pool import ConnectionPool
import os
# import sys
# import psycopg_pool
# from psycopg_pool import PoolTimeout
# from contextlib import asynccontextmanager
# from typing import AsyncIterator, Optional

host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'trader_bot')
db_user = os.getenv('DB_USER', 'postgres')
db_password = os.getenv('DB_PASSWORD', 'HkTAAtE2RD')


DATABASE_URL = "postgresql://{db_user}:{db_password}@{host}:{port}/{db_name}".format(host=host, port=port, db_user=db_user, db_password=db_password, db_name=db_name)

print(DATABASE_URL)

pool = ConnectionPool(DATABASE_URL, open=True, max_size=10)

def get_conn():
    return pool.getconn(timeout=120)



