#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import json
import os
import glob

# Parse CLI arguments
baseline_cm = 10.0
chessboard_cols = 9
chessboard_rows = 6
square_size_cm = 2.45
cam_left_id = 1
cam_right_id = 2
load_dir = None

for arg in sys.argv[1:]:
    if arg.startswith('-bl='):
        baseline_cm = float(arg.split('=')[1])
    elif arg.startswith('-cb='):
        cols, rows = arg.split('=')[1].split('x')
        chessboard_cols = int(cols)
        chessboard_rows = int(rows)
    elif arg.startswith('-sq='):
        square_size_cm = float(arg.split('=')[1])
    elif arg.startswith('-ci='):
        left, right = arg.split('=')[1].split(',')
        cam_left_id = int(left)
        cam_right_id = int(right)
    elif arg.startswith('-ld='):
        load_dir = arg.split('=')[1]
    elif arg == 'help':
        print('=== Stereo Camera Calibration ===')
        print('Arguments:')
        print('  -bl=NUM      Baseline distance in cm (default: 10)')
        print('  -cb=CxR      Chessboard inner corners (default: 9x6)')
        print('  -sq=NUM      Square size in cm (default: 2.45)')
        print('  -ci=L,R      Camera IDs (default: 1,2)')
        print('  -ld=DIR      Load images from directory (skips capture mode)')
        print()
        print('Directory mode expects files named: L01.png, R01.png, L02.png, R02.png, etc.')
        print()
        print('Live capture controls:')
        print('  SPACE - Capture calibration image')
        print('  ENTER - Process calibration and save')
        print('  ESC/Q - Quit')
        sys.exit(0)

print(f'=== Stereo Camera Calibration ===')
print(f'Baseline: {baseline_cm} cm')
print(f'Chessboard: {chessboard_cols}x{chessboard_rows} inner corners')
print(f'Square size: {square_size_cm} cm')

# Storage for calibration frames
calibration_frames_left = []
calibration_frames_right = []

# Load from directory mode
if load_dir is not None:
    print(f'Loading images from directory: {load_dir}')
    print()
    
    if not os.path.exists(load_dir):
        print(f'ERROR: Directory not found: {load_dir}')
        sys.exit(1)
    
    # Find all left images
    left_pattern = os.path.join(load_dir, 'L*.png')
    left_files = sorted(glob.glob(left_pattern))
    
    if len(left_files) == 0:
        print('ERROR: No left images found (pattern: L*.png)')
        sys.exit(1)
    
    print(f'Found {len(left_files)} left images')
    
    # Load corresponding right images
    for left_file in left_files:
        # Get the number from filename (e.g., L01.png -> 01)
        basename = os.path.basename(left_file)
        number = basename[1:].replace('.png', '')
        right_file = os.path.join(load_dir, f'R{number}.png')
        
        if not os.path.exists(right_file):
            print(f'WARNING: Skipping {basename} - no matching right image')
            continue
        
        # Load images
        frame_left = cv2.imread(left_file)
        frame_right = cv2.imread(right_file)
        
        if frame_left is None or frame_right is None:
            print(f'WARNING: Failed to load {basename}')
            continue
        
        calibration_frames_left.append(frame_left)
        calibration_frames_right.append(frame_right)
        print(f'Loaded pair: {basename} / R{number}.png')
    
    if len(calibration_frames_left) == 0:
        print('ERROR: No valid image pairs loaded')
        sys.exit(1)
    
    print(f'Loaded {len(calibration_frames_left)} image pairs')
    print()
    
    # Get dimensions from first frame
    height, width = calibration_frames_left[0].shape[:2]

# Live capture mode
else:
    print(f'Cameras: /dev/video{cam_left_id} and /dev/video{cam_right_id}')
    print()
    
    # Create output directory
    os.makedirs('captures', exist_ok=True)
    
    # Initialize cameras
    print('Connecting to cameras...')
    cam_left = cv2.VideoCapture(cam_left_id)
    cam_right = cv2.VideoCapture(cam_right_id)
    
    if not cam_left.isOpened() or not cam_right.isOpened():
        print('ERROR: Could not open cameras')
        sys.exit(1)
    
    cam_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get frame dimensions
    ret, frame = cam_left.read()
    if not ret:
        print('ERROR: Failed to read from camera')
        sys.exit(1)
    
    height, width = frame.shape[:2]
    print(f'Resolution: {width}x{height}')
    print()
    
    # Create window
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    
    print('=== Instructions ===')
    print('SPACE - Capture calibration image')
    print('ENTER - Process calibration and save')
    print('ESC/Q - Quit')
    print()
    print('Capture 10-20 images of the chessboard from different angles')
    print()
    
    # Main loop - capture calibration images
    while True:
        cam_left.grab()
        cam_right.grab()
        ret_left, frame_left = cam_left.retrieve()
        ret_right, frame_right = cam_right.retrieve()
        
        if not ret_left or not ret_right:
            print('ERROR: Failed to capture from cameras')
            break
        
        display_left = frame_left.copy()
        display_right = frame_right.copy()
        
        # Detect chessboard
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        ret_left_pattern, corners_left = cv2.findChessboardCorners(
            gray_left, (chessboard_cols, chessboard_rows), None)
        ret_right_pattern, corners_right = cv2.findChessboardCorners(
            gray_right, (chessboard_cols, chessboard_rows), None)
        
        # Draw corners if detected
        if ret_left_pattern:
            cv2.drawChessboardCorners(display_left, (chessboard_cols, chessboard_rows), 
                                     corners_left, ret_left_pattern)
        if ret_right_pattern:
            cv2.drawChessboardCorners(display_right, (chessboard_cols, chessboard_rows), 
                                     corners_right, ret_right_pattern)
        
        # Status display
        if ret_left_pattern and ret_right_pattern:
            status = "READY TO CAPTURE"
            color = (0, 255, 0)
        else:
            status = "NO PATTERN DETECTED"
            color = (0, 0, 255)
        
        cv2.putText(display_left, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_left, f"Captured: {len(calibration_frames_left)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show combined view
        combined = np.hstack((display_left, display_right))
        cv2.imshow('Calibration', combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Quit
        if key == ord('q') or key == 27:
            print('Calibration cancelled')
            cam_left.release()
            cam_right.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        
        # Capture frame
        elif key == 32:  # SPACE
            if ret_left_pattern and ret_right_pattern:
                calibration_frames_left.append(frame_left.copy())
                calibration_frames_right.append(frame_right.copy())
                frame_num = len(calibration_frames_left)
                cv2.imwrite(f"captures/L{frame_num:02d}.png", frame_left)
                cv2.imwrite(f"captures/R{frame_num:02d}.png", frame_right)
                print(f'Captured frame {frame_num}')
            else:
                print('Pattern not detected in both cameras')
        
        # Process calibration
        elif key == 13:  # ENTER
            if len(calibration_frames_left) < 5:
                print(f'Need at least 5 images. Currently have {len(calibration_frames_left)}')
                continue
            
            print()
            print('=== Processing Calibration ===')
            break
    
    # Release camera resources during processing
    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()

# Validate we have enough frames
if len(calibration_frames_left) < 5:
    print('ERROR: Need at least 5 calibration images')
    sys.exit(1)

print(f'Processing {len(calibration_frames_left)} image pairs...')
print()

# Prepare object points
objp = np.zeros((chessboard_cols * chessboard_rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)
objp *= (square_size_cm / 100.0)  # Convert to meters

obj_points = []
img_points_left = []
img_points_right = []

# Find chessboard corners in all captured frames
print('Finding chessboard corners...')
for i in range(len(calibration_frames_left)):
    gray_left = cv2.cvtColor(calibration_frames_left[i], cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(calibration_frames_right[i], cv2.COLOR_BGR2GRAY)
    
    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left, (chessboard_cols, chessboard_rows), None)
    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right, (chessboard_cols, chessboard_rows), None)
    
    if ret_left and ret_right:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        
        obj_points.append(objp)
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)
        print(f'Frame {i+1}: Valid')
    else:
        print(f'Frame {i+1}: Skipped')

if len(obj_points) < 5:
    print('ERROR: Not enough valid calibration images')
    sys.exit(1)

print(f'Using {len(obj_points)} valid image pairs')
print()

# Calibrate left camera
print('Calibrating left camera...')
ret_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    obj_points, img_points_left, (width, height), None, None)
print(f'Left camera error: {ret_left:.4f} pixels')

# Calibrate right camera
print('Calibrating right camera...')
ret_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    obj_points, img_points_right, (width, height), None, None)
print(f'Right camera error: {ret_right:.4f} pixels')
print()

# Stereo calibration
print('Performing stereo calibration...')
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_stereo, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, \
    R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        (width, height), criteria=criteria, flags=flags)

print(f'Stereo calibration error: {ret_stereo:.4f} pixels')

# Calculate baseline from calibration
calculated_baseline_cm = np.linalg.norm(T) * 100
print(f'Calculated baseline: {calculated_baseline_cm:.2f} cm')
print(f'Physical baseline: {baseline_cm} cm')
print()

# Stereo rectification
print('Computing rectification maps...')
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    (width, height), R, T, alpha=0)

# Generate undistort and rectify maps
map1_left, map2_left = cv2.initUndistortRectifyMap(
    camera_matrix_left, dist_coeffs_left, R1, P1, (width, height), cv2.CV_16SC2)

map1_right, map2_right = cv2.initUndistortRectifyMap(
    camera_matrix_right, dist_coeffs_right, R2, P2, (width, height), cv2.CV_16SC2)

# Prepare calibration data for saving
calibration_data = {
    'camera_matrix_left': camera_matrix_left.tolist(),
    'dist_coeffs_left': dist_coeffs_left.tolist(),
    'camera_matrix_right': camera_matrix_right.tolist(),
    'dist_coeffs_right': dist_coeffs_right.tolist(),
    'R': R.tolist(),
    'T': T.tolist(),
    'E': E.tolist(),
    'F': F.tolist(),
    'R1': R1.tolist(),
    'R2': R2.tolist(),
    'P1': P1.tolist(),
    'P2': P2.tolist(),
    'Q': Q.tolist(),
    'map1_left': map1_left.tolist(),
    'map2_left': map2_left.tolist(),
    'map1_right': map1_right.tolist(),
    'map2_right': map2_right.tolist(),
    'baseline_cm': baseline_cm,
    'calculated_baseline_cm': calculated_baseline_cm,
    'reprojection_error': ret_stereo,
    'image_width': width,
    'image_height': height,
    'chessboard_cols': chessboard_cols,
    'chessboard_rows': chessboard_rows,
    'square_size_cm': square_size_cm
}

# Save calibration
filename = 'stereo_calibration.json'
with open(filename, 'w') as f:
    json.dump(calibration_data, f, indent=2)

print(f'=== Calibration Complete ===')
print(f'Saved to: {filename}')
print(f'Reprojection error: {ret_stereo:.4f} pixels')
print(f'Valid calibration points: {len(obj_points) * len(objp)}')
print()
print('You can now use this calibration file in your main program')