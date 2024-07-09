"""
Author: Navjot Saroa
This file will collect data from the OpenF1 API to make the first layer of the database.
"""

import requests
import pandas as pd
import os

main_url = "https://api.openf1.org/v1/"
methods = ["car_data", "drivers", "intervals", "laps", "location", "meetings", "pit", "position", "race_control", "sessions", "stints", "team_radio", "weather"]

def dbmaker(url, methods):
    """
    Makes the database using the OpenF1 API
    param1: str url
    param2: str[] methods
    return: None
    """

    for method in methods:
        response = requests.get(url + method)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            filename = f"{method}.csv"

            if not os.path.exists(filename):
                df.to_csv(filename, index = False)

            print(df)

def lapsdbmaker(df):
    session_keys = df["session_key"]
    file = "laps.csv"
    progress = 0
    length = len(session_keys)

    for session_key in session_keys:
        print("Progress: ", progress)
        print(" Elements Remaining: ", length - progress)
        url = main_url + f"laps?session_key={session_key}"
        response = requests.get(url)
        data = pd.DataFrame(response.json())

        if not os.path.exists(file):
            data.to_csv(file)
        else:
            data.to_csv(file, mode='a', header = False)

        progress += 1


