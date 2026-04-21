# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:34:25 2026

@author: wyatt
"""

import cv2
import os

def vid_to_frame(input_name, fps = 5, memory_only=False):
    frames = []  # used only if saving in memory

    video_path = f"data/video_files/{input_name}.MP4"
    output_dir = f"data/frames/{input_name}"

    save_to_disk = f'data/frames/{input_name}'

    if save_to_disk:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    else:
        print(f"Video file opened successfully at {video_path}")
    
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Video's original frames per second: {video_fps}")
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video's original frame count: {total_frames}")
   
    frame_interval = max(1, int(round(video_fps / fps)))

    frame_count = 0
    saved_count = 0
    
    while True:
        success, image = vidcap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            
            if memory_only:
                frames.append(image)
                
            else:
                frame_filename = os.path.join(
                output_dir, f"frame_{saved_count:05d}.jpg")
    
                cv2.imwrite(frame_filename, image)
    
            saved_count += 1
        frame_count += 1

    vidcap.release()

    print("Video processing complete.")
    print(f"Saved {saved_count} frames at {fps} FPS.")
    
    if memory_only:
        return frames

