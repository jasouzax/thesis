#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import json
import os
import glob

# Parse CLI arguments
calib_num = None
cam_left_id = None
cam_right_id = None
num_disparities = 160
block_size = 15
save_output = False
output_fps = 10
show_info = True

for arg in sys.argv[1:]:
    if arg.startswith('-cn='):
        calib_num = int(arg.split('=')[1])
    elif arg.startswith('-ci='):
        left, right = arg.split('=')[1].split(',')
        cam_left_id = int(left)
        cam_right_id = int(right)
    elif arg.startswith('-nd='):
        num_disparities = int(arg.split('=')[1])
    elif arg.startswith('-bs='):
        block_size = int(arg.split('=')[1])
    elif arg.startswith('-save'):
        save_output = True
    elif arg.startswith('-fps='):
        output_fps = int(arg.split('=')[1])
    elif arg.startswith('-noinfo'):
        show_info = False
    elif arg == 'help':
        print('=== Stereo Depth Map Generator ===')
        print('Arguments:')
        print('  -cn=NUM      Calibration number to load (lists available if not specified)')
        print('  -ci=L,R      Camera IDs (default: from calibration file)')
        print('  -nd=NUM      Number of disparities, divisible by 16 (default: 160)')
        print('  -bs=SIZE     Block size, odd number 5-21 (default: 15)')
        print('  -save        Save output video as depth_output.avi')
        print('  -fps=NUM     Output video FPS when saving (default: 10)')
        print('  -noinfo      Hide calibration info overlay')
        print()
        print('Controls:')
        print('  Q/ESC - Quit')
        print('  D     - Toggle depth map display')
        print('  R     - Toggle rectified view')
        print('  I     - Toggle info overlay')
        print('  S     - Save current frame')
        sys.exit(0)

print('=== Stereo Depth Map Generator ===')
print()

# Find all calibration files
calib_pattern = 'stereo_calibration*.json'
calib_files = sorted(glob.glob(calib_pattern))

if len(calib_files) == 0:
    print('ERROR: No calibration files found')
    print('Run calibrate.py first to create calibration data')
    sys.exit(1)

# List available calibrations if -cn not specified
if calib_num is None:
    print('Available calibrations:')
    print()
    
    for idx, calib_file in enumerate(calib_files):
        try:
            with open(calib_file, 'r') as f:
                data = json.load(f)
            
            print(f'  [{idx}] {calib_file}')
            print(f'      Baseline: {data.get("baseline_cm", "N/A")} cm')
            print(f'      Resolution: {data.get("image_width", "N/A")}x{data.get("image_height", "N/A")}')
            print(f'      Error: {data.get("reprojection_error", "N/A"):.4f} pixels')
            print(f'      Chessboard: {data.get("chessboard_cols", "N/A")}x{data.get("chessboard_rows", "N/A")}')
            print()
        except:
            print(f'  [{idx}] {calib_file} (unreadable)')
            print()
    
    print('Usage: python3 depth.py -cn=NUM')
    print('Example: python3 depth.py -cn=0')
    sys.exit(0)

# Load selected calibration
if calib_num < 0 or calib_num >= len(calib_files):
    print(f'ERROR: Invalid calibration number {calib_num}')
    print(f'Available range: 0-{len(calib_files)-1}')
    sys.exit(1)

calib_file = calib_files[calib_num]
print(f'Loading: {calib_file}')
print()

with open(calib_file, 'r') as f:
    calib_data = json.load(f)

# Extract calibration parameters
camera_matrix_left = np.array(calib_data['camera_matrix_left'])
dist_coeffs_left = np.array(calib_data['dist_coeffs_left'])
camera_matrix_right = np.array(calib_data['camera_matrix_right'])
dist_coeffs_right = np.array(calib_data['dist_coeffs_right'])
R1 = np.array(calib_data['R1'])
R2 = np.array(calib_data['R2'])
P1 = np.array(calib_data['P1'])
P2 = np.array(calib_data['P2'])
Q = np.array(calib_data['Q'])

width = calib_data['image_width']
height = calib_data['image_height']
baseline_cm = calib_data['baseline_cm']
calculated_baseline_cm = calib_data.get('calculated_baseline_cm', 'N/A')
reproj_error = calib_data['reprojection_error']
chessboard_cols = calib_data.get('chessboard_cols', 'N/A')
chessboard_rows = calib_data.get('chessboard_rows', 'N/A')
square_size = calib_data.get('square_size_cm', 'N/A')

# Display calibration info
print('=== Calibration Details ===')
print(f'Resolution: {width}x{height}')
print(f'Physical baseline: {baseline_cm} cm')
print(f'Calculated baseline: {calculated_baseline_cm} cm' if isinstance(calculated_baseline_cm, float) else f'Calculated baseline: {calculated_baseline_cm}')
print(f'Reprojection error: {reproj_error:.4f} pixels')
print(f'Chessboard size: {chessboard_cols}x{chessboard_rows}')
print(f'Square size: {square_size} cm')
print()

# Compute rectification maps
print('Computing rectification maps...')
map1_left = np.array(calib_data['map1_left'], dtype=np.int16)
map2_left = np.array(calib_data['map2_left'], dtype=np.uint16)
map1_right = np.array(calib_data['map1_right'], dtype=np.int16)
map2_right = np.array(calib_data['map2_right'], dtype=np.uint16)

# Determine camera IDs
if cam_left_id is None or cam_right_id is None:
    cam_left_id = 1
    cam_right_id = 2
    print(f'Using default camera IDs: {cam_left_id}, {cam_right_id}')
else:
    print(f'Camera IDs: {cam_left_id}, {cam_right_id}')

# Initialize cameras
print('Connecting to cameras...')
cam_left = cv2.VideoCapture(cam_left_id)
cam_right = cv2.VideoCapture(cam_right_id)

if not cam_left.isOpened() or not cam_right.isOpened():
    print('ERROR: Could not open cameras')
    sys.exit(1)

cam_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Verify resolution
ret, frame = cam_left.read()
if not ret:
    print('ERROR: Failed to read from camera')
    sys.exit(1)

cam_height, cam_width = frame.shape[:2]
if cam_width != width or cam_height != height:
    print(f'WARNING: Camera resolution ({cam_width}x{cam_height}) differs from calibration')
    print(f'Resizing to calibration resolution: {width}x{height}')

# Create stereo matcher
print('Creating stereo matcher...')
print(f'Parameters: numDisparities={num_disparities}, blockSize={block_size}')
stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
stereo.setPreFilterCap(31)
stereo.setMinDisparity(0)
stereo.setUniquenessRatio(10)
stereo.setSpeckleWindowSize(100)
stereo.setSpeckleRange(32)
stereo.setDisp12MaxDiff(1)

# Create window
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)

# Video writer
video_writer = None
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('depth_output.avi', fourcc, output_fps, 
                                   (width * 2, height))
    print('Saving output to depth_output.avi')

# Display state
show_depth = True
show_rectified = False
frame_count = 0

print()
print('=== Controls ===')
print('Q/ESC - Quit')
print('D     - Toggle depth map')
print('R     - Toggle rectified view')
print('I     - Toggle info overlay')
print('S     - Save current frame')
print()
print('Running...')

# Main loop
while True:
    # Capture frames
    cam_left.grab()
    cam_right.grab()
    ret_left, frame_left = cam_left.retrieve()
    ret_right, frame_right = cam_right.retrieve()
    
    if not ret_left or not ret_right:
        print('ERROR: Failed to capture from cameras')
        break
    
    # Resize if needed
    if frame_left.shape[:2] != (height, width):
        frame_left = cv2.resize(frame_left, (width, height))
        frame_right = cv2.resize(frame_right, (width, height))
    
    # Rectify images
    rect_left = cv2.remap(frame_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_right, map2_right, cv2.INTER_LINEAR)
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # Choose display mode
    if show_rectified:
        display_left = rect_left.copy()
        display_right = rect_right.copy()
        
        # Draw horizontal lines
        for y in range(0, height, 30):
            cv2.line(display_left, (0, y), (width, y), (0, 255, 0), 1)
            cv2.line(display_right, (0, y), (width, y), (0, 255, 0), 1)
        
        mode_text = "RECTIFIED VIEW"
        
    elif show_depth:
        display_left = rect_left.copy()
        
        # Colorize disparity
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disparity_color = cv2.applyColorMap(disparity_normalized.astype(np.uint8), 
                                           cv2.COLORMAP_JET)
        display_right = disparity_color
        
        mode_text = "DEPTH MAP"
        
    else:
        display_left = frame_left.copy()
        display_right = frame_right.copy()
        mode_text = "RAW VIEW"
    
    # Add info overlay
    if show_info:
        info_y = 25
        line_height = 25
        
        cv2.putText(display_left, mode_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        info_y += line_height
        cv2.putText(display_left, f"Baseline: {baseline_cm} cm", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        info_y += line_height
        cv2.putText(display_left, f"Error: {reproj_error:.4f} px", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        info_y += line_height
        cv2.putText(display_left, f"Calib: [{calib_num}] {os.path.basename(calib_file)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Right side info
        cv2.putText(display_right, f"Frame: {frame_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if show_depth:
            cv2.putText(display_right, f"Disp: {num_disparities}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_right, f"Block: {block_size}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine views
    combined = np.hstack((display_left, display_right))
    cv2.imshow('Depth Map', combined)
    
    # Save to video
    if video_writer is not None:
        video_writer.write(combined)
    
    frame_count += 1
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:
        break
    elif key == ord('d'):
        show_depth = not show_depth
        show_rectified = False
        print(f'Depth map: {"ON" if show_depth else "OFF"}')
    elif key == ord('r'):
        show_rectified = not show_rectified
        show_depth = False
        print(f'Rectified view: {"ON" if show_rectified else "OFF"}')
    elif key == ord('i'):
        show_info = not show_info
        print(f'Info overlay: {"ON" if show_info else "OFF"}')
    elif key == ord('s'):
        cv2.imwrite(f'depth_frame_{frame_count:04d}.png', combined)
        print(f'Saved frame {frame_count}')

# Cleanup
print()
print('Cleaning up...')
cam_left.release()
cam_right.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

print(f'Processed {frame_count} frames')
print('Done!')