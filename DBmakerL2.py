"""
Author: Navjot Saroa
This file will collect data from the F1 game to make the second, and third layer of the database.
These databases will consist of general data collected across various sessions and conditions, third being more weekend specific than second.
"""
import sqlite3
from UDP import *

def init_db():
    conn = sqlite3.connect('packets.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS packets (
            id INTEGER PRIMARY KEY,
            m_sessionUID TEXT,
            m_packetId INTEGER,
            m_frameIdentifier INTEGER,
            packet BLOB
        )
    ''')
    conn.commit()
    return conn

def insert_packet(conn, packet_header, packet_bytes):
    c = conn.cursor()
    packet_id = packet_header.m_packetId
    session_uid = packet_header.m_sessionUID
    frame_identifier = packet_header.m_frameIdentifier

    c.execute('''
        INSERT INTO packets (m_sessionUID, m_packetId, m_frameIdentifier, packet)
        VALUES (?, ?, ?, ?)
    ''', (str(session_uid), packet_id, frame_identifier, packet_bytes))
    conn.commit()


def main(conn):
    file = open("../IPStuff.txt")       # Remove this line after making the relevant changes below
    UDP_IP = str(file.readline())[:-1]  # Replace with your own IP address in str format
    UDP_PORT = int(file.readline())     # Replace with your own port in int format
    
    connection = UDP(UDP_IP,UDP_PORT)
    sock = connection.connect()

    while True:
        data, addr = connection.receive(2048, sock)

        id = data.m_packetId
        handled_packet = connection.handle_packet(addr, connection.packet_type[id])
        header = handled_packet.m_header
        insert_packet(conn, header, handled_packet)

if __name__ == "__main__":
    conn = init_db()
    main(conn)
