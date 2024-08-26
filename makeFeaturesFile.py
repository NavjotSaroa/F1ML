import pandas as pd
import numpy as np
import sqlite3
from UDP import *

"""
Step 2 of process:
    Features used:
        Lap Data:
            m_currentLapTimeInMS
            m_lapDistance
            m_driverStatus
        Car Telemetry Data:
            m_brakesTemperature
            m_tyresSurfaceTemperature
            m_tyresInnerTemperature
        Car Status Data:
            m_fuelInTank
            m_actualTyreCompound
                Use this one to create a 1-hot system to identify compound
            m_tyresAgeLaps

    This project is a proof of concept since my machine does not have the storage space to create a large enough database for all tracks,
    so this project just focuses on Austria in dry conditions.

    Just a bit of relevant information:    
        Track length: 4318m

        T1:     466.66180419921875
        DRS1:   565.3001708984375
        T3:     1446.68896484375
        DRS2:   1523.369384765625
        T10:    4013.43017578125
        DRS3:   4144.26416015625
"""


def return_data(UDP_conn, data, packet_id):
    return UDP_conn.handle_packet(data, UDP_conn.packet_type[packet_id])


with open("../IPStuff.txt") as file:
    UDP_IP = str(file.readline())[:-1]  # Replace with your own IP address in str format
    UDP_PORT = int(file.readline())     # Replace with your own port in int format
UDP_conn = UDP(UDP_IP,UDP_PORT)

conn = sqlite3.connect("test.db")
cur = conn.cursor()
query = """
    SELECT * FROM packets;
"""
cur.execute(query)
rows = cur.fetchall()

laps = [row[3] for row in rows]
session_UIDs = [row[1] for row in rows]
lap_data = [row[4] for row in rows]
car_telemetry_data = [row[5] for row in rows]
car_status_data = [row[6] for row in rows]


# Initialise dictionary of features
features = {
    # Lap Data
    "m_lastLapTimeInMS":            [return_data(UDP_conn, data, 2).m_lapData[19].m_lastLapTimeInMS
                                    for data in lap_data
                                ],
    "m_currentLapTimeInMS":       [return_data(UDP_conn, data, 2).m_lapData[19].m_currentLapTimeInMS
                                    for data in lap_data
                                ],
    "m_lapDistance":              [return_data(UDP_conn, data, 2).m_lapData[19].m_lapDistance
                                    for data in lap_data
                                ],
    "m_driverStatus":             [return_data(UDP_conn, data, 2).m_lapData[19].m_driverStatus
                                    for data in lap_data
                                ],
    "m_totalDistance":          [return_data(UDP_conn, data, 2).m_lapData[19].m_totalDistance
                                    for data in lap_data
                                ],

    # Car Telemetry Data:
    "m_speed":                  [return_data(UDP_conn, data, 6).m_carTelemetryData[19].m_speed
                                    for data in car_telemetry_data
                                ],

    # Car Status Data
    "m_fuelInTank":               [return_data(UDP_conn, data, 7).m_carStatusData[19].m_fuelInTank
                                    for data in car_status_data
                                ],
    "isSoft":                     [int(return_data(UDP_conn, data, 7).m_carStatusData[19].m_actualTyreCompound == 16)
                                    for data in car_status_data
                                ],
    "isMed":                     [int(return_data(UDP_conn, data, 7).m_carStatusData[19].m_actualTyreCompound == 17)
                                    for data in car_status_data
                                ],
    "isHard":                     [int(return_data(UDP_conn, data, 7).m_carStatusData[19].m_actualTyreCompound == 18)
                                    for data in car_status_data
                                ],
    "m_tyresAgeLaps":               [return_data(UDP_conn, data, 7).m_carStatusData[19].m_tyresAgeLaps
                                    for data in car_status_data
                                ],

    # Misc:
    "lap":                      laps,
    "session_UID":              session_UIDs,
}

df_features = pd.DataFrame(features)


df_loaded = df_features                                     # Load df
df_loaded = df_loaded[df_loaded["m_driverStatus"] == 1]     # Only need rows for when driver is on track
df_loaded = df_loaded.drop(["m_driverStatus"], axis = 1)    # Remove the status column, they're all 1's now

reduced_groups = []
grouped = df_loaded.groupby(['session_UID', 'lap'])         # Group dataframe by UID and lap


# Iterate through each group
for name, group in grouped:
    # Only reduce the group if it has more than 3900 rows
    if len(group) > 3900:
        excess_rows = len(group) - 3900
        indices_to_drop = np.linspace(60, len(group) - 60, excess_rows, dtype=int)
        group = group.drop(group.index[indices_to_drop])
        reduced_groups.append(group)                        # Drops all laps that have less than 3900 rows, they were incomplete laps anyway

reduced_df = pd.concat(reduced_groups).reset_index(drop=True)

"""
All the stuff below is just to add those special distances to the dataframe, takes ages, could be made more efficient, but I only had to
use it once, and I am short on time for finishing this project
"""

reduced_df["speedAtT1"] = np.nan
reduced_df["speedAtDRS1"] = np.nan
reduced_df["speedAtT3"] = np.nan
reduced_df["speedAtDRS2"] = np.nan
reduced_df["speedAtT10"] = np.nan
reduced_df["speedAtDRS3"] = np.nan

for i in range(len(reduced_df)):
    df_3900 = reduced_df.iloc[i - 3900 : i]         # 3900 is the number of frames in a lap, we cut it down a few lines ago
    condition1 = df_3900[(df_3900["m_lapDistance"] >= (467 - 8)) & (df_3900["m_lapDistance"] <= 467)]       # Find all the frames in that distance range
    condition2 = df_3900[(df_3900["m_lapDistance"] >= (566 - 8)) & (df_3900["m_lapDistance"] <= 566)]       # I went with range because packets are sent based on time, not distance
    condition3 = df_3900[(df_3900["m_lapDistance"] >= (1447 - 8)) & (df_3900["m_lapDistance"] <= 1447)]     # Depending on how bad I am driving that lap, ths packet could come whenever and be off by a few metres
    condition4 = df_3900[(df_3900["m_lapDistance"] >= (1523 - 8)) & (df_3900["m_lapDistance"] <= 1523)]     # Packets are sent at 60Hz so it would only be off by a metre here and there
    condition5 = df_3900[(df_3900["m_lapDistance"] >= (4014 - 8)) & (df_3900["m_lapDistance"] <= 4014)]
    condition6 = df_3900[(df_3900["m_lapDistance"] >= (4145 - 8)) & (df_3900["m_lapDistance"] <= 4145)]

    if i%3900 == 0:
        print(condition1)
        print(condition2)
        print(condition3)
        print(condition4)
        print(condition5)
        print(condition6)
    if not condition1.empty:    # Append these frames to their respective lists
        last_valid_index1 = condition1.index[-1]
        speed_at_t1 = reduced_df.loc[last_valid_index1, 'm_speed']
        reduced_df.at[i, 'speedAtT1'] = speed_at_t1

    if not condition2.empty:
        last_valid_index2 = condition2.index[-1]
        speed_at_drs1 = reduced_df.loc[last_valid_index2, 'm_speed']
        reduced_df.at[i, 'speedAtDRS1'] = speed_at_drs1
        
    if not condition3.empty:
        last_valid_index3 = condition3.index[-1]
        speed_at_t3 = reduced_df.loc[last_valid_index3, 'm_speed']
        reduced_df.at[i, 'speedAtT3'] = speed_at_t3
        
    if not condition4.empty:
        last_valid_index4 = condition4.index[-1]
        speed_at_drs2 = reduced_df.loc[last_valid_index4, 'm_speed']
        reduced_df.at[i, 'speedAtDRS2'] = speed_at_drs2
        
    if not condition5.empty:
        last_valid_index5 = condition5.index[-1]
        speed_at_t10 = reduced_df.loc[last_valid_index5, 'm_speed']
        reduced_df.at[i, 'speedAtT10'] = speed_at_t10
        
    if not condition6.empty:
        last_valid_index6 = condition6.index[-1]
        speed_at_drs3 = reduced_df.loc[last_valid_index6, 'm_speed']
        reduced_df.at[i, 'speedAtDRS3'] = speed_at_drs3

        
reduced_df = reduced_df.iloc[3900:]
reduced_df.to_parquet('parquet_files/lap_data.parquet', engine='pyarrow', index=False)


