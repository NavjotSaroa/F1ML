"""
Author: Navjot Saroa
This file will collect data from the F1 game to make the second, third, and fourth layer of the database.
This database will consist of general data collected across various sessions and conditions.
"""

from UDP import *
import math
import pandas as pd
import os

file = open("../IPStuff.txt")       # Remove this line after making the relevant changes below
UDP_IP = str(file.readline())[:-1]  # Replace with your own IP address in str format
UDP_PORT = int(file.readline())     # Replace with your own port in int format

connection = UDP(UDP_IP,UDP_PORT)
sock = connection.connect()

broadcast_freq_hz = 20
damage_freq_hz = 10
slow_freq_hz = 2

gcd = math.gcd(broadcast_freq_hz, damage_freq_hz, slow_freq_hz)

counter = 0
index = 0
packets_per_cycle = int((3 * (broadcast_freq_hz / gcd)) + (damage_freq_hz / gcd) + (2 * (slow_freq_hz / gcd)))
# I havent bothered proving this yet but it seems like the gcd = 1/lcm when the two are calculated for products of powers of 2 and 5???
#
# The logic of packets_per_cycle is as follows:
#   session, and setups is broadcasted 2 times per second (slow_freq_hz)
#   damage is broadcasted 10 times per second (damage_freq_hz)
#   telemetry and status are broadcasted at the rate I have set in game (broadcast_freq_hz)

session_index = 0
setup_index = 0
telemetry_index = 0
status_index = 0
damage_index = 0
lap_index = 0

file = "layer2.csv"

fields = {
    "session": {
        "m_weather": [],
        "m_trackTemperature": [],
        "m_airTemperature": [],
        "m_totalLaps": [],
        "m_sessionType": [],
        "m_timeOfDay": [],
        "index": [],
    },
    "lap_data": {
        "m_lastLapTimeInMS": [],
        "m_currentLapTimeInMS": [],
        "m_sector1TimeInMS": [],
        "m_sector2TimeInMS": [],
        "index": [],
    },
    "setups": {
        "m_frontWing": [],
        "m_rearWing": [],
        "m_onThrottle": [],
        "m_offThrottle": [],
        "m_frontCamber": [],
        "m_rearCamber": [],
        "m_frontToe": [],
        "m_rearToe": [],
        "m_frontSuspension": [],
        "m_rearSuspension": [],
        "m_frontAntiRollBar": [],
        "m_rearAntiRollBar": [],
        "m_frontSuspensionHeight": [],
        "m_rearSuspensionHeight": [],
        "m_brakePressure": [],
        "m_brakeBias": [],
        "m_rearLeftTyrePressure": [],
        "m_rearRightTyrePressure": [],
        "m_frontLeftTyrePressure": [],
        "m_frontRightTyrePressure": [],
        "m_ballast": [],
        "m_fuelLoad": [],
        "index": []
    },
    "telemetry": {
        "m_brakesTemperature": [],
        "m_tyresSurfaceTemperature": [],
        "m_tyresInnerTemperature": [],
        "m_engineTemperature": [],
        "m_tyresPressure": [],
        "index": []
    },
    "status": {
        "m_fuelInTank": [],
        "m_actualTyreCompound": [],
        "m_tyreAgeLaps": [],
        "index": []
    },
    "damage": {
        "m_tyresWear": [],
        "index": []
    }
}

while True:
    header, data = connection.receive(2048, sock)


    id = header.m_packetId
    handled_packet = connection.handle_packet(data, connection.packet_type[id])
    packet_time = handled_packet.m_header.m_sessionTime

    """
    Session: 1 (2 Hz)
    LapData: 2 (20 Hz)
    CarSetups: 5 (2 Hz)
    CarTelemetry: 6 (20 Hz)
    CarStatus: 7 (20 Hz)
    CarDamage: 10 (10 Hz)
    """

    if id == 1:
        for _ in range(10):
            fields["session"]["m_weather"].append(handled_packet.m_weather)
            fields["session"]["m_trackTemperature"].append(handled_packet.m_trackTemperature)
            fields["session"]["m_airTemperature"].append(handled_packet.m_airTemperature)
            fields["session"]["m_totalLaps"].append(handled_packet.m_totalLaps)
            fields["session"]["m_timeOfDay"].append(handled_packet.m_timeOfDay)
            fields["session"]["m_sessionType"].append(handled_packet.m_sessionType)
            fields["session"]["index"].append(session_index)
            session_index += 1
    elif id == 2:
        fields["lap_data"]["m_lastLapTimeInMS"].append(handled_packet.m_lapData[0].m_lastLapTimeInMS)
        fields["lap_data"]["m_currentLapTimeInMS"].append(handled_packet.m_lapData[0].m_currentLapTimeInMS)
        fields["lap_data"]["m_sector1TimeInMS"].append(handled_packet.m_lapData[0].m_sector1TimeInMS)
        fields["lap_data"]["m_sector2TimeInMS"].append(handled_packet.m_lapData[0].m_sector2TimeInMS)
        fields["lap_data"]["index"].append(lap_index)
        lap_index += 1
    elif id == 5:
        for _ in range(10):
            fields["setups"]["m_frontWing"].append(handled_packet.m_carSetups[0].m_frontWing)
            fields["setups"]["m_rearWing"].append(handled_packet.m_carSetups[0].m_rearWing)
            fields["setups"]["m_onThrottle"].append(handled_packet.m_carSetups[0].m_onThrottle)
            fields["setups"]["m_offThrottle"].append(handled_packet.m_carSetups[0].m_offThrottle)
            fields["setups"]["m_frontCamber"].append(handled_packet.m_carSetups[0].m_frontCamber)
            fields["setups"]["m_rearCamber"].append(handled_packet.m_carSetups[0].m_rearCamber)
            fields["setups"]["m_frontToe"].append(handled_packet.m_carSetups[0].m_frontToe)
            fields["setups"]["m_rearToe"].append(handled_packet.m_carSetups[0].m_rearToe)
            fields["setups"]["m_frontSuspension"].append(handled_packet.m_carSetups[0].m_frontSuspension)
            fields["setups"]["m_rearSuspension"].append(handled_packet.m_carSetups[0].m_rearSuspension)
            fields["setups"]["m_frontAntiRollBar"].append(handled_packet.m_carSetups[0].m_frontAntiRollBar)
            fields["setups"]["m_rearAntiRollBar"].append(handled_packet.m_carSetups[0].m_rearAntiRollBar)
            fields["setups"]["m_frontSuspensionHeight"].append(handled_packet.m_carSetups[0].m_frontSuspensionHeight)
            fields["setups"]["m_rearSuspensionHeight"].append(handled_packet.m_carSetups[0].m_rearSuspensionHeight)
            fields["setups"]["m_brakePressure"].append(handled_packet.m_carSetups[0].m_brakePressure)
            fields["setups"]["m_brakeBias"].append(handled_packet.m_carSetups[0].m_brakeBias)
            fields["setups"]["m_rearLeftTyrePressure"].append(handled_packet.m_carSetups[0].m_rearLeftTyrePressure)
            fields["setups"]["m_rearRightTyrePressure"].append(handled_packet.m_carSetups[0].m_rearRightTyrePressure)
            fields["setups"]["m_frontLeftTyrePressure"].append(handled_packet.m_carSetups[0].m_frontLeftTyrePressure)
            fields["setups"]["m_frontRightTyrePressure"].append(handled_packet.m_carSetups[0].m_frontRightTyrePressure)
            fields["setups"]["m_ballast"].append(handled_packet.m_carSetups[0].m_ballast)
            fields["setups"]["m_fuelLoad"].append(handled_packet.m_carSetups[0].m_fuelLoad)
            fields["setups"]["index"].append(setup_index)
            setup_index += 1
    elif id == 6:
        brake_temp = handled_packet.m_carTelemetryData[0].m_brakesTemperature
        tyre_surface_temp = handled_packet.m_carTelemetryData[0].m_tyresSurfaceTemperature
        tyre_inner_temp = handled_packet.m_carTelemetryData[0].m_tyresInnerTemperature
        tyre_pressure = handled_packet.m_carTelemetryData[0].m_tyresPressure

        fields["telemetry"]["m_brakesTemperature"].append(tuple(brake_temp))
        fields["telemetry"]["m_tyresSurfaceTemperature"].append(tuple(tyre_surface_temp))
        fields["telemetry"]["m_tyresInnerTemperature"].append(tuple(tyre_inner_temp))
        fields["telemetry"]["m_engineTemperature"].append(handled_packet.m_carTelemetryData[0].m_engineTemperature)
        fields["telemetry"]["m_tyresPressure"].append(tuple(tyre_pressure))
        fields["telemetry"]["index"].append(telemetry_index)
        telemetry_index += 1
    elif id == 7:
        fields["status"]["m_fuelInTank"].append(handled_packet.m_carStatusData[0].m_fuelInTank)
        fields["status"]["m_actualTyreCompound"].append(handled_packet.m_carStatusData[0].m_actualTyreCompound)
        fields["status"]["m_tyreAgeLaps"].append(handled_packet.m_carStatusData[0].m_tyresAgeLaps)
        fields["status"]["index"].append(status_index)
        status_index += 1
    elif id == 10:
        for _ in range(2):
            damage = handled_packet.m_carDamageData[0].m_tyresWear
            fields["damage"]["m_tyresWear"].append(tuple(damage))
            fields["damage"]["index"].append(damage_index)
            damage_index += 1


    if id in [1, 2, 5, 6, 7, 10]:
        counter += 1

    if counter == packets_per_cycle:
        counter = 0
        combined_dict = {}
        for outerkey in fields.keys():
            for innerkey in fields[outerkey].keys():
                combined_dict[f"{outerkey}_{innerkey}"] = fields[outerkey][innerkey]

        print(combined_dict)
        try:
            df = pd.DataFrame(combined_dict)

            print(df)

            if not os.path.exists(file):
                df.to_csv(file)
            else:
                existing_df = pd.read_csv(file)
                starting_index = existing_df.index[-1] + 1

                df.index = range(starting_index, starting_index + len(df))
                df.to_csv(file, mode='a', header = False)

        except ValueError:
            pass

        fields = {
            "session": {
                "m_weather": [],
                "m_trackTemperature": [],
                "m_airTemperature": [],
                "m_totalLaps": [],
                "m_sessionType": [],
                "m_timeOfDay": [],
                "index": [],
            },
            "lap_data": {
                "m_lastLapTimeInMS": [],
                "m_currentLapTimeInMS": [],
                "m_sector1TimeInMS": [],
                "m_sector2TimeInMS": [],
                "index": [],
            },
            "setups": {
                "m_frontWing": [],
                "m_rearWing": [],
                "m_onThrottle": [],
                "m_offThrottle": [],
                "m_frontCamber": [],
                "m_rearCamber": [],
                "m_frontToe": [],
                "m_rearToe": [],
                "m_frontSuspension": [],
                "m_rearSuspension": [],
                "m_frontAntiRollBar": [],
                "m_rearAntiRollBar": [],
                "m_frontSuspensionHeight": [],
                "m_rearSuspensionHeight": [],
                "m_brakePressure": [],
                "m_brakeBias": [],
                "m_rearLeftTyrePressure": [],
                "m_rearRightTyrePressure": [],
                "m_frontLeftTyrePressure": [],
                "m_frontRightTyrePressure": [],
                "m_ballast": [],
                "m_fuelLoad": [],
                "index": []
            },
            "telemetry": {
                "m_brakesTemperature": [],
                "m_tyresSurfaceTemperature": [],
                "m_tyresInnerTemperature": [],
                "m_engineTemperature": [],
                "m_tyresPressure": [],
                "index": []
            },
            "status": {
                "m_fuelInTank": [],
                "m_actualTyreCompound": [],
                "m_tyreAgeLaps": [],
                "index": []
            },
            "damage": {
                "m_tyresWear": [],
                "index": []
            }
        }