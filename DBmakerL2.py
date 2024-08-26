"""
Author: Navjot Saroa

Step 1 of process:
    This file is supposed to act as a replacement to DBMakerL2 and dataSplitter, I
    want to streamline the code so that it takes in the UDP telemetry data packets
    and also sort them at the same time, making sure that i dont have multiple
    databases storing the same thing in multiple ways, which is inefficient.
"""

import sqlite3
from UDP import *

class LiveDBMaker():
    def __init__(self, ip_file_path, db_file, batch_size = 1000, required_packets = [2, 6, 7]):
        # UDP stuff
        self.UDP_conn = self.get_udp_connection(ip_file_path)
        self.sock = self.UDP_conn.connect()

        # Sqlite3 stuff
        self.sqlite_conn = self.get_sqlite_connection(db_file)
        
        # Other important stuff
        self.batch = []
        self.batch_size = batch_size                        # Number of frames all data is received from before committing to the database
        self.required_packets = required_packets            
        self.packet_names = [
            self.UDP_conn.packet_type[packet_id].__name__
            for packet_id in self.required_packets
        ]

    # UDP connection
    def get_udp_connection(self, ip_file_path):
        """Initialize the UDP connection based on the IP and port from a file."""
        with open(ip_file_path) as file:
            UDP_IP = str(file.readline())[:-1]  # Replace with your own IP address in str format
            UDP_PORT = int(file.readline())     # Replace with your own port in int format
        return UDP(UDP_IP, UDP_PORT)


    # Sqlite3 connection
    def get_sqlite_connection(self, db_file):
        """Initialize the SQLite connection and create the required table if it does not exists."""
        conn = sqlite3.connect(db_file)
        # Create table if it doesn't already exist
        query = """
        CREATE TABLE IF NOT EXISTS packets (
            id INTEGER PRIMARY KEY,
            m_sessionUID TEXT,
            m_frameIdentifier INTEGER,
            lap INTEGER,
            packetLapData BLOB,
            packetCarTelemetryData BLOB,
            packetCarStatusData BLOB
        )
        """
        conn.execute(query)
        conn.commit()
        return conn

    def commit_batch(self):
        """
        Commits a batch of rows to the database.
        
        Parameters:
            sqlite_conn: SQLite database connection
            batch: List of rows to be committed
            packet_names: Dynamic packet column names to be inserted
        """
        cursor = self.sqlite_conn.cursor()

        # Join the dynamic column names into the query
        dynamic_columns = ', '.join(self.packet_names)
        placeholders = ', '.join(['?' for _ in self.packet_names])

        query = f"""
        INSERT INTO packets (m_sessionUID, m_frameIdentifier, lap, {dynamic_columns})
        VALUES (?, ?, ?, {placeholders})
        """

        cursor.executemany(query, self.batch)    # Insert the batch into the database
        self.sqlite_conn.commit()

    def collect_packets(self):
        """Collects the data packets sent by the game and picks out the ones that are useful.
        Then it processes it and sticks it into the database
        
        Parameters:
            UDP_conn: Connection to the UDP
        """

        ref_fingerprint = (None, None)  # To track sessionUID and frameIdentifier

        try:
            while True:
                # Receive and handle incoming UDP data packet

                data, addr = self.UDP_conn.receive(2048, self.sock)   # Gets the data packet from the game, packet has not been handled yet, just so happens that its header can be accessed without that
                packet_id = data.m_packetId
                driver_id = data.m_playerCarIndex

                if packet_id in self.required_packets:
                    handled_packet = self.UDP_conn.handle_packet(addr, self.UDP_conn.packet_type[packet_id])
                    
                    session_uid = data.m_sessionUID
                    frame_identifier = data.m_frameIdentifier
                    fingerprint = (str(session_uid), frame_identifier) # The int value of session_uid is too large so we store as string


                    # I am going to create a list generator, if the fingerprint of the newly received packet matches the fingerprint of the last element of the list
                    # it will insert right away. Otherwise, it will clear the list and start a new one. If i don't have all the packets from a certain frame,
                    # that entire frame is useless to me.

                    #  WARNING: This system only works under the assumption that packet 2 will come first, change this is you decide you need other packets
                    if fingerprint == ref_fingerprint:                              # Check fingerprint, will only return True for packets 6 or 7
                        row.append(handled_packet)                                  # Add to row
                        if len(row) == len(self.required_packets) + 3:              # Account for sessionUID, frameIdentifier, lap
                            self.batch.append(tuple(row))                           # Send away to batch if row if full
                            row = []                                                # Clear out the row, at this point all of 2, 6, and 7 had been in the row so we can do this
                            
                    elif packet_id == 2:
                        row = [
                            fingerprint[0], fingerprint[1], 
                            handled_packet.m_lapData[driver_id].m_currentLapNum, 
                            handled_packet
                        ]                                                           # Clear out the row, or start a new one. Either way, don't need that previous data anymore
                        ref_fingerprint = (str(session_uid), frame_identifier)      # Create a new fingerprint

                    if len(self.batch) >= self.batch_size:                          # If batch size is reached, commit the batch to the database
                        self.commit_batch()
                        self.batch.clear()                                          # Clear batch after commit

        except Exception as e:
            print(f"Error in collect_packets: {e}")    
    
    def locate_points(self):
        """
        Extra function to locate car's current position on track by pressing the L3 button on the PS5
        """
        print("locating")
        pack_3_received = False
        try:
            while True:
                # Receive and handle incoming UDP data packet
                data, addr = self.UDP_conn.receive(2048, self.sock)   # Gets the data packet from the game, packet has not been handled yet, just so happens that its header can be accessed without that
                packet_id = data.m_packetId
                driver_id = data.m_playerCarIndex
                if packet_id == 3:
                    handled_packet = self.UDP_conn.handle_packet(addr, self.UDP_conn.packet_type[packet_id])

                    deets = bytes(handled_packet.m_eventDetails)
                    bitflags = int.from_bytes(deets[:-8], byteorder='little')

                    # Print the decoded text
                    if bitflags == 0x2000:
                        pack_3_received = True                         # We only want to check the distance if we have asked for it
                    else:
                        continue

                if packet_id == 2 and pack_3_received == True:
                    handled_packet = self.UDP_conn.handle_packet(addr, self.UDP_conn.packet_type[packet_id])
                    lap_distance = handled_packet.m_lapData[driver_id].m_lapDistance
                    print(lap_distance)                                 # This number is how far from the start line you are in the lap
                    pack_3_received = False

        except Exception as e:
            print(f"Error in collect_packets: {e}") 


if __name__ == "__main__":
    thing = LiveDBMaker("../IPStuff.txt", "databases/test.db")
    thing.locate_points()