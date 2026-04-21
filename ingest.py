# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 01:03:19 2026

@author: wyatt
"""

import argparse
import os

from db import get_connection, init_db
from vid_to_frame import vid_to_frame
from court_detection import create_homography
from yolo_player_detection import process_video

# want to be able to handle if a video already has data written to the database, can either be stopped or overwritten
def get_or_create_video(cursor, video_name, overwrite=False):
    cursor.execute(
        "SELECT video_id FROM videos WHERE video_name = ?",
        (video_name,)
        )
    row = cursor.fetchone()
    
    if row:
        video_id = row[0]
        
        if overwrite:
            cursor.execute(
                "DELETE FROM tracking_data WHERE video_id = ?",
                (video_id,)
                )
            return video_id
        
        else:
            raise ValueError(
                f" Video '{video_name}' exists. Overwrite it with --overwrite")
            
    cursor.execute(
        "INSERT INTO videos (video_name) VALUES (?)",
        (video_name,)
        )
    return cursor.lastrowid

# tracking data into the database!
def insert_tracking_data(cursor, video_id, data):
    rows = [
        (
            video_id,
            d["frame"],
            d["time_sec"],
            float(d["real_x"]),
            float(d["real_y"]),
            float(d["adj_x"]),
            float(d["speed_mph"]),
            float(d["total_distance"])
            )
        for d in data
        ]
    
    cursor.executemany("""
                       INSERT INTO tracking_data (
                           video_id, frame, time_sec, real_x, real_y, adj_x, speed_mph, total_distance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       """, rows)

def ensure_frames(input_name, fps):
    frame_dir = f"data/frames/{input_name}"
    
    if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 0:
        print("Frames exist already")
    else:
        print("Extracting frames")
        vid_to_frame(input_name, fps=fps)
        
def ensure_homography(input_name):
    homography_path = f"homography/homography_{input_name}.npy"
    
    if os.path.exists(homography_path):
        print("Homography exists!")
    else:
        print("Create homography...")
        create_homography(input_name)
        
def main():
    print("This is starting!")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input_name", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--save_csv", action="store_true")
    
    args = parser.parse_args()
    print ("I made it past parsing args!")
    init_db()
    print ("I made it past initializing the database!")    
    the_name = args.input_name
    print ("I know the video name!")
    # frame extraction check
    ensure_frames(the_name, args.fps)
    print ("I've checked to see if the video exists in the datase!")
    # homography check
    ensure_homography(the_name)
    print ("I've checked to see if homography already exists!'")
    # process the video
    print("Player detection running")
    
    # get the data!
    data = process_video(the_name, save_csv=args.save_csv)
    print(f"Processed {len(data)} frames")
    
    # save to database
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        print("Writing to database")
        video_id = get_or_create_video(cursor, args.input_name, args.overwrite)

        insert_tracking_data(cursor, video_id, data)

        conn.commit()
        print("\nPipeline completed successfully!")

    except Exception as e:
        conn.rollback()
        print("\n[ERROR]", e)

    finally:
        conn.close()
        
#%%

if __name__ == "__main__":
    main()