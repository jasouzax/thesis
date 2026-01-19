import cv2
import numpy as np
import json
import os
import time

# --- CONFIGURATION ---
CAMERA_ID = 0            
CHECKERBOARD = (9, 6)    
MIN_FRAMES = 15          
OUTPUT_FILE = "flatten.json"
STATIONARY_TIME = 5.0    
MOVEMENT_THRESHOLD = 3.0 

# Global Trackbar Params (Initial Defaults)
THRESH_BLOCK_SIZE = 21 # Must be odd
THRESH_C = 5
MORPH_KERNEL = 0       # 0 = Off

def nothing(x):
    pass

def make_binary(image, block_size, c_val, morph_k):
    """
    Converts image to strictly Black and White using Adaptive Thresholding.
    Uses dynamic parameters.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Ensure block_size is odd and >= 3
    if block_size % 2 == 0: block_size += 1
    if block_size < 3: block_size = 3

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, block_size, c_val)
    
    # Morphological Operations (remove noise)
    if morph_k > 0:
        k_size = morph_k
        kernel = np.ones((k_size, k_size), np.uint8)
        # Morph Open: Erosions followed by Dilation (removes small white noise)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary

def calculate_homography_error(objp_2d, corners):
    if corners is None or len(corners) != len(objp_2d):
        return 9999.0
    corners_2d = corners.reshape(-1, 2)
    H, mask = cv2.findHomography(objp_2d, corners_2d, cv2.RANSAC, 5.0)
    if H is None: return 9999.0
    projected_corners = cv2.perspectiveTransform(objp_2d.reshape(-1, 1, 2), H).reshape(-1, 2)
    error = np.linalg.norm(corners_2d - projected_corners, axis=1).mean()
    return error

def calibrate_lens():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp_2d = objp[:, :2]

    objpoints = [] 
    imgpoints = [] 

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    # Create Window and Trackbars
    cv2.namedWindow('Flatten Binary Tuner', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Block Size', 'Flatten Binary Tuner', THRESH_BLOCK_SIZE, 99, nothing)
    cv2.createTrackbar('C Constant', 'Flatten Binary Tuner', THRESH_C, 50, nothing)
    cv2.createTrackbar('Noise Filter', 'Flatten Binary Tuner', MORPH_KERNEL, 10, nothing)

    print(f"--- FLAT BINARY TUNER ---")
    print(f"Adjust trackbars to make the checkerboard clear and stable.")
    print(f"Goal: Sharp black squares, no noise dots.")

    count = 0
    auto_mode = False
    last_corners = None
    stable_start_time = None
    best_frame_data = None 
    min_error_in_window = 9999.0

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        h, w = frame.shape[:2]
        should_save = False
        saved_corners = None
        
        # Read Trackbars
        blk = cv2.getTrackbarPos('Block Size', 'Flatten Binary Tuner')
        c_val = cv2.getTrackbarPos('C Constant', 'Flatten Binary Tuner')
        morph = cv2.getTrackbarPos('Noise Filter', 'Flatten Binary Tuner')

        # --- BINARIZE ---
        binary_frame = make_binary(frame, blk, c_val, morph)
        
        # Find corners on the BINARY image
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret_corners, corners_raw = cv2.findChessboardCorners(binary_frame, CHECKERBOARD, flags)

        display_frame = frame.copy()
        
        if ret_corners:
            # Subpixel refine on binary
            corners = cv2.cornerSubPix(binary_frame, corners_raw, (11, 11), (-1, -1), 
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # Visuals
            vis_binary = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_binary, CHECKERBOARD, corners, ret_corners)
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)
            
            # --- Auto Mode Logic ---
            if auto_mode:
                if last_corners is not None:
                    movement = np.linalg.norm(corners - last_corners, axis=2).mean()
                    if movement < MOVEMENT_THRESHOLD:
                        current_time = time.time()
                        if stable_start_time is None:
                            stable_start_time = current_time
                            best_frame_data = None
                            min_error_in_window = 9999.0

                        error = calculate_homography_error(objp_2d, corners)
                        if error < min_error_in_window:
                            min_error_in_window = error
                            best_frame_data = (frame.copy(), corners.copy(), error)
                            
                        elapsed = current_time - stable_start_time
                        countdown = max(0, STATIONARY_TIME - elapsed)
                        bar_width = int((elapsed / STATIONARY_TIME) * 200)
                        cv2.rectangle(display_frame, (30, 90), (30 + bar_width, 105), (0, 255, 0), -1)
                        cv2.rectangle(display_frame, (30, 90), (230, 105), (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Stable: {countdown:.1f}s", (240, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        if elapsed >= STATIONARY_TIME:
                            should_save = True
                            stable_start_time = None
                    else:
                        stable_start_time = None
                        best_frame_data = None
                        cv2.putText(display_frame, "Moving...", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    stable_start_time = None
                last_corners = corners
            else:
                last_corners = None
                stable_start_time = None
            
            if not should_save:
                cv2.putText(display_frame, "Detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if best_frame_data is not None:
                    # saved_corners = best_frame_data[1] # Actually, let's just use current to be safe with display
                    saved_corners = corners # Simpler flow
                else:
                    saved_corners = corners
        else:
            vis_binary = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)

        # OSD
        mode_text = "AUTO: ON" if auto_mode else "AUTO: OFF (Press 'a')"
        cv2.putText(display_frame, mode_text, (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if auto_mode else (0,0,255), 2)
        cv2.putText(display_frame, f"Saved: {count}/{MIN_FRAMES}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # SHOW BOTH VIEWS
        disp_small = cv2.resize(display_frame, (480, 360))
        bin_small = cv2.resize(vis_binary, (480, 360))
        combined = np.hstack((disp_small, bin_small))
        cv2.imshow('Flatten Binary Tuner', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            auto_mode = not auto_mode
            stable_start_time = None
        if (key == ord('s') or should_save) and ret_corners:
            if not should_save: saved_corners = corners
            if saved_corners is not None:
                objpoints.append(objp)
                imgpoints.append(saved_corners)
                count += 1
                print(f"Frame {count} captured.")
                stable_start_time = None
                best_frame_data = None
                cv2.waitKey(100)
        elif key == ord('q'):
            if count < MIN_FRAMES: print(f"Need {MIN_FRAMES - count} more.")
            else: break

    cap.release()
    cv2.destroyAllWindows()

    if count >= MIN_FRAMES:
        print("Calibrating...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        data = { "camera_matrix": mtx.tolist(), "dist_coeffs": dist.tolist(), "width": w, "height": h, "reprojection_error": ret }
        with open(OUTPUT_FILE, 'w') as f: json.dump(data, f, indent=4)
        print(f"Saved to '{OUTPUT_FILE}'. Error: {ret:.4f}")

if __name__ == "__main__":
    calibrate_lens()
