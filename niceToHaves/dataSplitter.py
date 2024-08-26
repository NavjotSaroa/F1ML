"""
Author: Navjot Saroa
Will take the sqlite3 data set created in DBmakerL2.py and will split it into groups by m_packetId.
A choice had to be made weather to make the database deeper at the cost of speed, and I chose against it.
The packages being sent are little databases of their own and can be accessed just by handling the BLOB type data in these tables.
"""

import sqlite3
from UDP import *
import ctypes as ct


class TableMaker():
    def __init__(self, db_name):
        self.db_name = db_name
        
    def make_connection(self):
        conn = sqlite3.connect(f"{self.db_name}.db")
        return conn

    def init_tables(self, conn):
        """Creates a new table for each data packet type"""
        c = conn.cursor()
        table_names = [cls.__name__ for cls in UDP.packet_type.values()]
        classes = UDP.packet_type.values()
        
        tuples = (list(zip(table_names, classes)))
        tuples = [tuples[i] for i in [2, 6, 7]]
        print(tuples)
        for name, cls in tuples:
            base_query = """CREATE TABLE IF NOT EXISTS """
            base_query += name
            base_query += """ ( 
            id INTEGER PRIMARY KEY, 
            m_sessionUID TEXT,
            m_packetId INTEGER,
            m_frameIdentifier INTEGER,"""
            fields = cls._fields_

            for field in fields:
                base_query += f"{field[0]} {self.get_basic_ctype_name(field[1])},\n"
            base_query = base_query[:-2]
            base_query += ")"
            print(base_query)
            # c.execute(base_query)
            # conn.commit()

    def init_cleaned(self, conn):
        """Creates a cleaned table that only consists of 3 packets and that is it: PacketLapData, PacketCarStatusData, PacketCarTelemetryData.
        This is the one that should be used when making the quick database for the ML work, I am leaving the other one in since that is useful
        for creating a massive dataset if need be in the future."""
        file = open("../IPStuff.txt")       # Remove this line after making the relevant changes below
        UDP_IP = str(file.readline())[:-1]  # Replace with your own IP address in str format
        UDP_PORT = int(file.readline())     # Replace with your own port in int format
        
        connection = UDP(UDP_IP,UDP_PORT)
        sock = connection.connect()


        c = conn.cursor
        table_names = [cls.__name__ for cls in UDP.packet_type.values()]
        classes = UDP.packet_type.values()
        tuples = (list(zip(table_names, classes)))
        tuples = [tuples[i] for i in [2, 6, 7]]
        print(tuples)

        c = conn.cursor()
        c.execute("SELECT * FROM packets")
        rows = c.fetchall()


        for row in rows:
            primary_key = row[0]
            print(primary_key)
        #     # print(primary_key)
        #     id = row[2]
        #     handled_packet = connection.handle_packet(row[4], connection.packet_type[id])
        #     attributes = [getattr(handled_packet, field[0]) for field in handled_packet._fields_]
        #     attributes = [primary_key, handled_packet.m_header.m_sessionUID, handled_packet.m_header.m_packetId, handled_packet.m_header.m_frameIdentifier] + attributes
        #     for index in range(len(attributes)):
        #         if not isinstance(attributes[index], (int, float, str)):
        #                 attributes[index] = bytes(attributes[index])

        #     attributes[1] = str(attributes[1])
        #     values = tuple(attributes)
        #     columns = self.get_table_columns(conn, connection.packet_type[id].__name__)     # Requires the existence of the table made in init_tables
        #     print(columns)


        base_query = """CREATE TABLE IF NOT EXISTS grouped_packets"""
        # for name, cls in tuples:



    def sort_root(self, conn):

        file = open("../IPStuff.txt")       # Remove this line after making the relevant changes below
        UDP_IP = str(file.readline())[:-1]  # Replace with your own IP address in str format
        UDP_PORT = int(file.readline())     # Replace with your own port in int format
        
        connection = UDP(UDP_IP,UDP_PORT)
        sock = connection.connect()

        c = conn.cursor()
        c.execute("SELECT * FROM packets")
        rows = c.fetchall()

        
        for row in rows:
            primary_key = row[0]
            # print(primary_key)
            id = row[2]
            handled_packet = connection.handle_packet(row[4], connection.packet_type[id])
            attributes = [getattr(handled_packet, field[0]) for field in handled_packet._fields_]
            attributes = [primary_key, handled_packet.m_header.m_sessionUID, handled_packet.m_header.m_packetId, handled_packet.m_header.m_frameIdentifier] + attributes
            for index in range(len(attributes)):
                if not isinstance(attributes[index], (int, float, str)):
                        attributes[index] = bytes(attributes[index])

            attributes[1] = str(attributes[1])
            values = tuple(attributes)
            columns = self.get_table_columns(conn, connection.packet_type[id].__name__)     # Requires the existence of the table made in init_tables


            placeholders = ', '.join(['?' for _ in columns])
            columns_str = ', '.join(columns)

            query = f"INSERT INTO {connection.packet_type[id].__name__} ({columns_str}) VALUES ({placeholders})"
            try:
                c.execute(query, values)
                conn.commit()
            except sqlite3.IntegrityError as e:
                pass
        conn.close()
            

    def get_packet_by_id(self, conn, id):
        c = conn.cursor()
        c.execute('''
            SELECT * FROM packets WHERE id = ?
        ''', (id,))
        result = c.fetchone()
        if result:
            return result
        return None
    
    def drop_table(self, conn, table_name):
        c = conn.cursor()
        c.execute(f'''
        DROP TABLE {table_name}
        ''')
        conn.commit()
        conn.close()

    
    def get_basic_ctype_name(self, field):
        if issubclass(field, (ct.c_int, ct.c_uint, ct.c_long, ct.c_ulong, ct.c_short, ct.c_ushort, ct.c_byte, ct.c_ubyte, ct.c_longlong, ct.c_ulonglong)):
            return "INTEGER"
        elif issubclass(field, (ct.c_float, ct.c_double)):
            return "REAL"
        elif issubclass(field, ct.c_char_p):
            return "TEXT"
        else:
            return "BLOB"
        
    def get_table_columns(self, conn, table_name):
        c = conn.cursor()
        c.execute(f'PRAGMA table_info({table_name})')
        columns = c.fetchall()
        return [column[1] for column in columns]  # Return list of column names



if __name__ == "__main__":
    root = TableMaker("packets")
    conn = root.make_connection()
    
    root.init_cleaned(conn)
    
    
    
    
    
    
    
    
    # root.init_tables(conn)
