from strats.follow import FollowSignal, FollowTrader
from util.update import get_all_updates
from datetime import datetime
import os


"""
Last Run: 2025-04-28
PID: xyz
"""


def start_algorithm(algorithm: any, **kwargs):
    print()
    get_all_updates()
    print(f"Initializing [{algorithm.__name__()}] | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | PID: {os.getpid()}")
    algorithm.init_algorithm()
    print(f"Starting     [{algorithm.__name__()}] | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | PID: {os.getpid()}")
    algorithm.start(**kwargs)


if __name__ == "__main__":
    algo = FollowTrader(
        ticker="kxhighny",
        date="realtime",
        signal=FollowSignal(metric="result"),
        rsa_private_key=os.getenv("KALSHI_API_KEY"),
        api_key_id=os.getenv("KALSHI_API_KEY_ID"),
        data_dir="../data",
    )
    start_algorithm(algo)
