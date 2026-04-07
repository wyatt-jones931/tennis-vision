# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 01:35:16 2026

@author: wyatt
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

input_name = "CornerFloorView"
frame_dir = f"data/frames/{input_name}" 
output_dir = f"data/output_frames/{input_name}"  
video_name = f"data/output_video_files/{input_name}.mp4"
os.makedirs(output_dir, exist_ok=True)

# Prep the model for player detection
model = YOLO("yolov8n.pt")

# Load homography matrix to convert image space to real space
H = np.load("homography.npy")

# Constants
CONF_THRESHOLD = 0.5
NET_Y = 39

COURT_WIDTH = 36
COURT_LENGTH = 78
FULL_WIDTH = 60
FULL_LENGTH = 120

RUNOFF_X = 12
RUNOFF_Y = 21

FPS_extracted_frames = 5
DT = 1/FPS_extracted_frames

prev_position = None
prev_speed = None
total_distance = 0

data = []

# MAIN SET OF FUNCTIONS
def get_closest_player(result):
    candidates = []

    for box in result.boxes:
        if int(box.cls[0]) == 0 and box.conf[0] > CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_y = (y1 + y2) // 2
            candidates.append((center_y, (x1, y1, x2, y2)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# Convets the image/pixel space to real using homography matrix
def pixel_to_real(px, py, H): 
    point = np.array([[[px, py]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    return transformed[0][0]  # (x, y) in feet

def get_real_position(box, H):
    x1, y1, x2, y2 = box
    px = (x1 + x2) / 2
    py = y2
    return pixel_to_real(px, py, H)

def to_full_coordinates(real_x, real_y):
    full_x = np.clip(real_x + RUNOFF_X, 0, FULL_WIDTH)
    full_y = np.clip(real_y + RUNOFF_Y, 0, FULL_LENGTH)
    return [full_x, full_y]

def update_motion(prev_pos, current_pos, total_dist, prev_speed, dt):
    if prev_pos is None:
        return current_pos, total_dist, 0.0, 0.0  # speed, smoothed_speed

    # Distance (feet)
    dist = np.linalg.norm(current_pos - prev_pos)
    total_dist += dist

    if dist < 0.1:  # ignore tiny movements (~1 inch)
        return prev_pos, total_dist, 0.0, prev_speed if prev_speed else 0.0

    # Instantaneous speed (ft/s)
    speed = dist / dt

    # Smooth speed
    ALPHA = 0.5
    if prev_speed is None:
        smoothed_speed = speed
    else:
        smoothed_speed = ALPHA * prev_speed + (1 - ALPHA) * speed

    return current_pos, total_dist, speed, smoothed_speed

# MINIMAP STUFF
# Keep track of all player positions for the minimap
player_positions = []

# Keep track of player positions in feet
player_positions_centered = []

# Mini-map size (pixels)
minimap_width, minimap_height = 200, 400

def draw_minimap(current_pos, history):
    
    # Mini-map canvas
    minimap = np.zeros((minimap_height, minimap_width, 3), dtype=np.uint8)

    # Semi-transparent background
    overlay = minimap.copy()
    overlay[:] = (30, 30, 30)  # dark grey
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, minimap, 1 - alpha, 0, minimap)

    net_y = COURT_LENGTH / 2
    service_line_dist = 21

    # Center court in mini-map
    offset_x = (FULL_WIDTH - COURT_WIDTH) / 2
    offset_y = (FULL_LENGTH - COURT_LENGTH) / 2

    # Scaling
    scale_x = minimap_width / FULL_WIDTH
    scale_y = minimap_height / FULL_LENGTH

    # Convert full play coordinates to pixels
    def to_pixel(pos):
        x = int(pos[0] * scale_x)
        y = minimap_height - int(pos[1] * scale_y)  # flip Y so baseline side is bottom
        return x, y

    # Draw court lines (relative to full play coordinates)
    def draw_lines():
        cv2.rectangle(minimap,
                      to_pixel([offset_x, offset_y]),
                      to_pixel([offset_x + COURT_WIDTH, offset_y + COURT_LENGTH]),
                      (0, 255, 0), 2)

        net_y_px = offset_y + net_y
        # Net
        cv2.line(minimap,
                 to_pixel([offset_x, net_y_px]),
                 to_pixel([offset_x + COURT_WIDTH, net_y_px]),
                 (0, 255, 0), 2)

        # Service lines
        cv2.line(minimap,
                 to_pixel([offset_x, net_y_px + service_line_dist]),
                 to_pixel([offset_x + COURT_WIDTH, net_y_px + service_line_dist]),
                 (0, 255, 0), 1)
        cv2.line(minimap,
                 to_pixel([offset_x, net_y_px - service_line_dist]),
                 to_pixel([offset_x + COURT_WIDTH, net_y_px - service_line_dist]),
                 (0, 255, 0), 1)

        # Center service line
        cv2.line(minimap,
                 to_pixel([offset_x + COURT_WIDTH / 2, net_y_px - service_line_dist]),
                 to_pixel([offset_x + COURT_WIDTH / 2, net_y_px + service_line_dist]),
                 (0, 255, 0), 1)

    draw_lines()

    # Draw trail
    for i, pos in enumerate(history):
        alpha = i / len(history)
        color = (0, 0, int(255 * alpha))
        x, y = to_pixel(pos)
        cv2.circle(minimap, (x, y), 3, color, -1)

    if current_pos is not None:
        x, y = to_pixel(current_pos)
        cv2.circle(minimap, (x, y), 6, (0, 0, 255), -1)
        
    return minimap

prev_position = None
total_distance = 0

frame_files = sorted(os.listdir(frame_dir))

frame_idx = 0

# MAJOR LOOP
for filename in frame_files:
    frame = cv2.imread(os.path.join(frame_dir, filename))
    
    # Get time of real video for CSV file
    time_sec = frame_idx / FPS_extracted_frames

    results = model(frame)
    box = get_closest_player(results[0])

    if box is not None:
        real_x, real_y = get_real_position(box, H)
        
        adj_x = real_x - (COURT_WIDTH / 2)
        adj_y = real_y

        player_positions_centered.append([adj_x, adj_y])

        if real_y <= NET_Y:
            current_position = np.array([real_x, real_y])
        
            prev_position, total_distance, speed, smoothed_speed = update_motion(
                prev_position,
                current_position,
                total_distance,
                prev_speed,
                DT
            )
        
            prev_speed = smoothed_speed  # update for next frame
            
            # Stuf for CSV
            speed_mph = smoothed_speed * 0.681818
            
            # Writing to CSV
            data.append({
                "frame": frame_idx,
                "time_sec": time_sec,
                "real_x": real_x,
                "real_y": real_y,
                "adj_x": adj_x,
                "speed_ft_s": smoothed_speed,
                "speed_mph": speed_mph,
                "total_distance": total_distance
            })
        
            # Convert to minimap coordinates
            full_position = to_full_coordinates(real_x, real_y)
            player_positions.append(full_position)
        
            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
            text_x = x1
            text_y = y1 - 10  # 10 pixels above top-left of bbox
        
            # Display distance
            cv2.putText(frame,
                        f"Distance: {total_distance:.1f} ft",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2)
        
            # Display speed (mph)
            speed_mph = smoothed_speed * 0.681818
            cv2.putText(frame,
                        f"Speed: {speed_mph:.1f} mph",
                        (text_x, text_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2)

    # Only draw minimap if we have a valid position
    if full_position is not None:
        minimap = draw_minimap(full_position, player_positions)

        h, w, _ = frame.shape
        x_offset = w - minimap_width - 10
        y_offset = 10

        roi = frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width]
        cv2.addWeighted(minimap, 0.7, roi, 0.3, 0, roi)
        frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width] = roi
    
    cv2.imwrite(os.path.join(output_dir, filename), frame)
    frame_idx += 1
    
# Make the video version
images = sorted(os.listdir(output_dir))
frame_sample = cv2.imread(os.path.join(output_dir, images[0]))
height, width, _ = frame_sample.shape

video = cv2.VideoWriter(
    video_name,
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (width, height)
)

for image in images:
    img_path = os.path.join(output_dir, image)
    video.write(cv2.imread(img_path))

video.release()
print(f"Video saved as {video_name}")

df = pd.DataFrame(data)
df.to_csv(f"data/{input_name}_tracking.csv", index=False)