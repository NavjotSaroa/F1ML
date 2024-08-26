"""
Author: Navjot Saroa
This file will receive data from the F1 game and graph all the relevant car data live
"""

import matplotlib.animation as animation
import threading
from UDP import UDP
from structures import *
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

def receive_data():
    while True:
        header, data = connection.receive(2048, sock)

        if header.m_packetId == 6:
            packTele = connection.handle_packet(data, connection.packet_type[header.m_packetId])

        else:
            continue
        
        carData = packTele.m_carTelemetryData[0]
        speed = carData.m_speed
        throttle = carData.m_throttle
        steer = carData.m_steer
        brake = carData.m_brake
        clutch = carData.m_clutch
        gear = carData.m_gear
        rpm = carData.m_engineRPM
        brakeTempRL = carData.m_brakesTemperature[0]
        brakeTempRR = carData.m_brakesTemperature[1]
        brakeTempFL = carData.m_brakesTemperature[2]
        brakeTempFR = carData.m_brakesTemperature[3]  
        tyreInnerTempRL = carData.m_tyresInnerTemperature[0]  
        tyreInnerTempRR = carData.m_tyresInnerTemperature[1]  
        tyreInnerTempFL = carData.m_tyresInnerTemperature[2]  
        tyreInnerTempFR = carData.m_tyresInnerTemperature[3]
        drs = carData.m_drs  

        info["speed"].append(speed)
        if len(info["speed"]) > 3000:
            info["speed"].pop(0)

        info["throttle"].append(throttle)
        if len(info["throttle"]) > 3000:
            info["throttle"].pop(0)

        info["steer"].append(steer)
        if len(info["steer"]) > 3000:
            info["steer"].pop(0)

        info["brake"].append(brake)
        if len(info["brake"]) > 3000:
            info["brake"].pop(0)

        info["clutch"].append(clutch)
        if len(info["clutch"]) > 3000:
            info["clutch"].pop(0)

        info["gear"].append(gear)
        if len(info["gear"]) > 3000:
            info["gear"].pop(0)

        info["rpm"].append(rpm)
        if len(info["rpm"]) > 3000:
            info["rpm"].pop(0)

        info["brakeTempRL"].append(brakeTempRL)
        if len(info["brakeTempRL"]) > 3000:
            info["brakeTempRL"].pop(0)

        info["brakeTempRR"].append(brakeTempRR)
        if len(info["brakeTempRR"]) > 3000:
            info["brakeTempRR"].pop(0)

        info["brakeTempFL"].append(brakeTempFL)
        if len(info["brakeTempFL"]) > 3000:
            info["brakeTempFL"].pop(0)

        info["brakeTempFR"].append(brakeTempFR)
        if len(info["brakeTempFR"]) > 3000:
            info["brakeTempFR"].pop(0)

        info["tyreInnerTempRL"].append(tyreInnerTempRL)
        if len(info["tyreInnerTempRL"]) > 3000:
            info["tyreInnerTempRL"].pop(0)

        info["tyreInnerTempRR"].append(tyreInnerTempRR)
        if len(info["tyreInnerTempRR"]) > 3000:
            info["tyreInnerTempRR"].pop(0)

        info["tyreInnerTempFL"].append(tyreInnerTempFL)
        if len(info["tyreInnerTempFL"]) > 3000:
            info["tyreInnerTempFL"].pop(0)

        info["tyreInnerTempFR"].append(tyreInnerTempFR)
        if len(info["tyreInnerTempFR"]) > 3000:
            info["tyreInnerTempFR"].pop(0)

        info["drs"].append(drs)
        if len(info["drs"]) > 3000:
            info["drs"].pop(0)

if __name__ == "__main__":
    file = open("../IPStuff.txt")
    # Set up the UDP server
    UDP_IP = str(file.readline())[:-1]
    UDP_PORT = int(file.readline())

    connection = UDP(UDP_IP, UDP_PORT)

    sock = connection.connect()
    print(f"Listening on {UDP_IP}:{UDP_PORT}")

    info = {
        "speed": [],
        "throttle": [],
        "steer": [],
        "brake": [],
        "clutch": [],
        "gear": [],
        "rpm": [],
        "brakeTempRL": [],
        "brakeTempRR": [],
        "brakeTempFL": [],
        "brakeTempFR": [],
        "tyreInnerTempRL": [],
        "tyreInnerTempRR": [],
        "tyreInnerTempFL": [],
        "tyreInnerTempFR": [],
        "drs": [],
    }

    receiver_thread = threading.Thread(target = receive_data)
    receiver_thread.daemon = True
    receiver_thread.start()
    lines_glow = {}  # Initialize the lines_glow dictionary
    fills = {key: None for key in info.keys()}

    def update_plot(frames, line, data_key):
        artists = [line]
        if info[data_key]:
            xdata = np.arange(len(info[data_key]))
            ydata = info[data_key]
            line.set_ydata(ydata)
            line.set_xdata(xdata)
            
            # Remove previous fill_between collection if it exists
            if fills[data_key] is not None:
                fills[data_key].remove()
            
            # Create a new fill_between collection
            fills[data_key] = axes[data_key].fill_between(xdata, ydata, color=line.get_color(), alpha=0.3)
            artists.append(fills[data_key])
            
            axes[data_key].relim()
            axes[data_key].autoscale_view()
        
        return artists

    # Set up the initial plot
    fig, ax = plt.subplots(4,4)
    axes = {
        'speed': ax[0, 0],
        'throttle': ax[0, 1],
        'steer': ax[0, 2],
        'brake': ax[0,3],
        'clutch': ax[1, 0],
        'gear': ax[1, 1],
        'rpm': ax[1,2],
        'drs': ax[1,3],
        'brakeTempRL': ax[2, 0],
        'brakeTempRR': ax[2, 1],
        'brakeTempFL': ax[2, 2],
        'brakeTempFR': ax[2, 3],
        'tyreInnerTempRL': ax[3, 0],
        'tyreInnerTempRR': ax[3, 1],
        'tyreInnerTempFL': ax[3, 2],
        'tyreInnerTempFR': ax[3, 3],
    }

    fig.patch.set_facecolor('#212946')
    for key, ax in axes.items():
        ax.set_facecolor('#212946')
        ax.grid(color='#2A3459')
        ax.tick_params(colors='#D3D3D3')  # Change color of ticks to light gray
        ax.spines['bottom'].set_color('#D3D3D3')
        ax.spines['top'].set_color('#D3D3D3') 
        ax.spines['left'].set_color('#D3D3D3')
        ax.spines['right'].set_color('#D3D3D3')
        ax.title.set_color('#D3D3D3')  # Change title color to light gray

    # Define line colors for each data key
    line_colors = {
        'speed': '#08F7FE',
        'throttle': '#FE53BB',
        'steer': '#F5D300',
        'brake': '#00FF41',
        'clutch': '#FF8C00',
        'gear': '#FFD700',
        'rpm': '#FF69B4',
        'brakeTempRL': '#00CED1',
        'brakeTempRR': '#00CED1',
        'brakeTempFL': '#00CED1',
        'brakeTempFR': '#00CED1',
        'tyreInnerTempRL': '#FF4500',
        'tyreInnerTempRR': '#FF4500',
        'tyreInnerTempFL': '#FF4500',
        'tyreInnerTempFR': '#FF4500',
        'drs': '#800080',

    }

    # Create lines with initial setup
    lines = {}
    for key in axes:
        lines[key] = axes[key].plot([], [], lw=2, color=line_colors[key])[0]

    axes['speed'].set_ylim(0, 400)
    axes['throttle'].set_ylim(0, 1)
    axes['steer'].set_ylim(-1, 1)
    axes['brake'].set_ylim(0, 1)
    axes['clutch'].set_ylim(0, 1)
    axes['gear'].set_ylim(0, 8)
    axes['rpm'].set_ylim(0, 15000)
    axes['brakeTempRL'].set_ylim(0, 1000)
    axes['brakeTempRR'].set_ylim(0, 1000)
    axes['brakeTempFL'].set_ylim(0, 1000)
    axes['brakeTempFR'].set_ylim(0, 1000)  
    axes['tyreInnerTempRL'].set_ylim(90, 110)
    axes['tyreInnerTempRR'].set_ylim(90, 110)
    axes['tyreInnerTempFL'].set_ylim(90, 110)
    axes['tyreInnerTempFR'].set_ylim(90, 110)

    for key in axes:
        axes[key].set_xlim(0, 3000)
        axes[key].set_title(key.capitalize())
        axes[key].grid(True)

    # Use FuncAnimation to update each plot
    animations = []
    for key in lines:
        ani = animation.FuncAnimation(fig, update_plot, fargs=(lines[key], key), blit=True)
        animations.append(ani)

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.show()