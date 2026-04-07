import os
import cv2
import numpy as np
from ultralytics import YOLO

input_name = "BaselineElevatedView"
frame_dir = f"data/frames/{input_name}" 
output_dir = f"data/output_frames/{input_name}"  
video_name = f"data/output_video_files/{input_name}.mp4"
os.makedirs(output_dir, exist_ok=True)

# Prep the model for player detection
model = YOLO("yolov8n.pt")

# Load homography matrix to convert image space to real space
H = np.load("homography.npy")

NET_Y = 39  # half court (feet)

# Convets the image/pixel space to real using homography matrix
def pixel_to_real(px, py, H): 
    point = np.array([[[px, py]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    return transformed[0][0]  # (x, y) in feet

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

    # Full play dimensions
    total_width, total_length = 60, 120  # ft
    court_width, court_length = 36, 78
    net_y = court_length / 2
    service_line_dist = 21

    # Center court in mini-map
    offset_x = (total_width - court_width) / 2
    offset_y = (total_length - court_length) / 2

    # Scaling
    scale_x = minimap_width / total_width
    scale_y = minimap_height / total_length

    # Convert full play coordinates to pixels
    def to_pixel(pos):
        x = int(pos[0] * scale_x)
        y = minimap_height - int(pos[1] * scale_y)  # flip Y so baseline side is bottom
        return x, y

    # Draw court lines (relative to full play coordinates)
    def draw_lines():
        cv2.rectangle(minimap,
                      to_pixel([offset_x, offset_y]),
                      to_pixel([offset_x + court_width, offset_y + court_length]),
                      (0, 255, 0), 2)

        net_y_px = offset_y + net_y
        # Net
        cv2.line(minimap,
                 to_pixel([offset_x, net_y_px]),
                 to_pixel([offset_x + court_width, net_y_px]),
                 (0, 255, 0), 2)

        # Service lines
        cv2.line(minimap,
                 to_pixel([offset_x, net_y_px + service_line_dist]),
                 to_pixel([offset_x + court_width, net_y_px + service_line_dist]),
                 (0, 255, 0), 1)
        cv2.line(minimap,
                 to_pixel([offset_x, net_y_px - service_line_dist]),
                 to_pixel([offset_x + court_width, net_y_px - service_line_dist]),
                 (0, 255, 0), 1)

        # Center service line
        cv2.line(minimap,
                 to_pixel([offset_x + court_width / 2, net_y_px - service_line_dist]),
                 to_pixel([offset_x + court_width / 2, net_y_px + service_line_dist]),
                 (0, 255, 0), 1)

    draw_lines()

    # Draw trail
    for pos in history:
        x, y = to_pixel(pos)
        cv2.circle(minimap, (x, y), 3, (0, 0, 255), -1)

    # Draw current player
    x, y = to_pixel(current_pos)
    cv2.circle(minimap, (x, y), 6, (0, 0, 255), -1)

    return minimap

prev_position = None
total_distance = 0

frame_files = sorted(os.listdir(frame_dir))

for filename in frame_files:
    frame_path = os.path.join(frame_dir, filename)
    frame = cv2.imread(frame_path)

    results = model(frame)
    result = results[0]

    player_candidates = []

    # Collect all person detections
    for box in result.boxes:
        if int(box.cls[0]) == 0 and box.conf[0] > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_y = (y1 + y2) // 2
            player_candidates.append((center_y, (x1, y1, x2, y2)))

    closest_player_box = None

    if len(player_candidates) > 0:
        # Sort by vertical position (largest y = closest)
        player_candidates.sort(key=lambda x: x[0], reverse=True)
        closest_player_box = player_candidates[0][1]

    if closest_player_box is not None:
        x1, y1, x2, y2 = closest_player_box
        
        # Get foot position (bottom center of bbox)
        px = (x1 + x2) / 2
        py = y2
    
        # Convert to real-world coordinates
        real_x, real_y = pixel_to_real(px, py, H)
    
        # Shift origin to baseline center
        adj_x = real_x - 18
        adj_y = real_y
        
        player_positions_centered.append([adj_x, adj_y])
        
        norm_x = adj_x / 18    # -1 to 1
        norm_y = adj_y / 39    # 0 to 1
        
        # Filter to near side only
        if real_y <= NET_Y:
    
            # Track movement
            current_position = np.array([real_x, real_y])
    
            if prev_position is not None:
                dist = np.linalg.norm(current_position - prev_position)
                total_distance += dist
    
            prev_position = current_position
            
            # Convert court coordinates → full play coordinates
            runoff_x = 12
            runoff_y = 21
            
            full_x = real_x + runoff_x
            full_y = real_y + runoff_y
            
            # Clamp so it never goes off-map
            full_x = np.clip(full_x, 0, 60)
            full_y = np.clip(full_y, 0, 120)
            
            full_position = [full_x, full_y]
            player_positions.append(full_position)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
            # Display real-world position
            cv2.putText(frame,
                        f"{adj_x:.1f}ft, {adj_y:.1f}ft",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)
            
            # Display total distance
            cv2.putText(frame,
                        f"Distance: {total_distance:.1f} ft",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2)

            # Put frame around tracked player in video
            cv2.putText(frame, "Tracked Player",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)
        
    # Draw minimap
    minimap = draw_minimap(full_position, player_positions)
    
    # Overlay on top-right corner
    h, w, _ = frame.shape
    x_offset = w - minimap_width - 10
    y_offset = 10  # top corner
    
    # Alpha blending for placing on frame
    roi = frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width]
    cv2.addWeighted(minimap, 0.7, roi, 0.3, 0, roi)
    frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width] = roi
   
    # Save frame
    cv2.imwrite(os.path.join(output_dir, filename), frame)

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
