import pandas as pd
import sqlite3
from UDP import *
from matplotlib import pyplot as plt
from scipy.stats import kendalltau

def packet_for_query(packet_name):
    query = f"""
    WITH RankedEntries AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY m_sessionUID, m_currentLapNum ORDER BY ROWID) as rn
        FROM
            cleaned_packets
        WHERE
            {packet_name} IS NOT NULL AND CarStatusData IS NOT NULL
    )
    SELECT
        m_sessionUID,
        m_currentLapNum,
        m_lastLapTimeInMS,
        m_sector1TimeInMS,
        m_sector2TimeInMS,
        m_currentLapTimeInMS,
        MotionData,
        SessionData,
        LapData,
        EventData,
        ParticipantsData,
        CarSetupData,
        CarTelemetryData,
        CarStatusData,
        FinalClassificationData,
        LobbyInfoData,
        CarDamageData,
        SessionHistoryData,
        TyreSetsData,
        MotionExData
    FROM
        RankedEntries
    WHERE
        rn = 1;
    """

    return query

query = packet_for_query("SessionData")
conn = sqlite3.connect("cleaned.db")
cur = conn.cursor()

# Execute the query
cur.execute(query)

# Fetch and print the results
results = cur.fetchall()
sessions = [row[0] for row in results]
sessions = list(set(sessions))

tyres_wear = [UDP.handle_packet(row[7], UDP.packet_type[1]).m_trackTemperature for row in results if row[2] >= 50000 and row[2] <= 120000 ]
times = [row[2] for row in results if row[2] >= 50000 and row[2] <= 120000]


corr, p = kendalltau(tyres_wear, times)
print(corr, p)
plt.scatter(tyres_wear, times)
plt.show()

