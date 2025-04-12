import sys
from utils.download import download_all
from services.daily import decide
from datetime import datetime
import warnings
from db import get_conn
warnings.filterwarnings('ignore')


def switch_case(case):
    switch = {
        "download": download,
        "decide": decide_handler,
    }
    return switch.get(case, default_case)()


def default_case():
    print("Invalid case. Please provide a valid arg.")

def download():
    download_all()

def decide_handler():
    today = datetime.now()
    conn = get_conn()
    decide(today, conn)
    conn.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    switch_case(args[-1])

