"""
Author: Navjot Saroa
This file will collect data from the OpenF1 API to make the first layer of the database.
I know the idea of first making a CSV and then a sqlite3 database from it seems inefficient. I came up with the sqlite3 idea later and I have too much on my plate to fix this issue right now.
"""

import requests
import pandas as pd
import os
import sqlite3

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

def csv_to_sqlite(csv_files, database_name):
    # Create a connection to the SQLite3 database
    conn = sqlite3.connect(database_name)
    
    for csv_file in csv_files:
        # Extract the table name from the CSV file name
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Load the CSV file into a pandas dataframe
        df = pd.read_csv(csv_file)
        
        # Write the dataframe to the SQLite3 database
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        
        print(f"Table '{table_name}' created successfully.")
    
    # Close the connection
    conn.close()


if __name__ == "__main__":
    folder_path = "layer1"
    csv_files = []

    # Iterate through all the files in the specified directory
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            # Construct the full file path and add it to the list
            full_path = os.path.join(folder_path, file_name)
            csv_files.append(full_path)
    database_name = "layer1.db"

    csv_to_sqlite(csv_files, database_name)