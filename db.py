# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:50:18 2026

@author: wyatt
"""

import sqlite3

DB_NAME = "tennis.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_df():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tracking_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER,
        frame INTEGER,
        time_sec REAL,
        real_x REAL,
        real_y REAL,
        adj_x REAL,
        speed_mph REAL,
        total_distance REAL,
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    )
    """)

    conn.commit()
    conn.close()