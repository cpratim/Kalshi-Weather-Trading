from datetime import datetime
from time import sleep
import os

if __name__ == "__main__":

    while True:
        if datetime.now().hour >= 8:
            os.system("python start.py &")
            print(f"Started algorithm at {datetime.now()}")
            break

        sleep(60)