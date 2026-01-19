import cv2
import numpy as np
import json
import os
import time

# --- CONFIGURATION ---
CAMERA_ID = 0            # Change to 1 or 2 if you have multiple cameras
CHECKERBOARD = (9, 6)    # Internal corners (rows-1, columns-1)
MIN_FRAMES = 15          # Minimum frames needed for calibration
OUTPUT_FILE = "flatten.json"
STATIONARY_TIME = 5.0    # Seconds to hold still
MOVEMENT_THRESHOLD = 3.0 # Max pixel movement to be considered stable

def calculate_homography_error(objp_2d, corners):
    """
    Calculates the reprojection error assuming a planar homography.
    This checks how well the detected corners fit a perfect plane projection.
    """
    if corners is None or len(corners) != len(objp_2d):
        return 9999.0
    
    # corners dim is (N, 1, 2), objp_2d is (N, 2)
    corners_2d = corners.reshape(-1, 2)
    
    # Find Homography
    H, mask = cv2.findHomography(objp_2d, corners_2d, cv2.RANSAC, 5.0)
    if H is None:
        return 9999.0
        
    # Project object points
    projected_corners = cv2.perspectiveTransform(objp_2d.reshape(-1, 1, 2), H).reshape(-1, 2)
    
    # Calculate error (average euclidian distance)
    # This roughly creates a "pixel error" metric for the single frame
    error = np.linalg.norm(corners_2d - projected_corners, axis=1).mean()
    return error

def calibrate_lens():
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    # This defines the "ideal" flat world coordinates
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # 2D version for Homography check (drop Z)
    objp_2d = objp[:, :2]

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    print(f"--- F L A T T E N   M A X   C A L I B R A T I O N ---")
    print(f"1. Show the checkerboard to the camera.")
    print(f"2. Press 's' to save a frame when corners are detected.")
    print(f"3. Press 'a' to toggle AUTO MODE (finds BEST frame in 5s window).")
    print(f"4. Need at least {MIN_FRAMES} frames. Press 'q' to finish.")

    count = 0
    
    # Auto-snap state variables
    auto_mode = False
    last_corners = None
    stable_start_time = None
    
    # Best Frame Buffer
    best_frame_data = None # (frame, corners, error)
    min_error_in_window = 9999.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w = frame.shape[:2]
        should_save = False
        saved_corners = None # To pass out if needed

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret_corners, corners_raw = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        # Draw corners for visual feedback
        display_frame = frame.copy()
        
        if ret_corners:
            # Subpixel refine immediately for error calc
            corners = cv2.cornerSubPix(gray, corners_raw, (11, 11), (-1, -1), 
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)
            
            # --- Auto Mode Logic ---
            if auto_mode:
                if last_corners is not None:
                    # Calculate movement (average Euclidean distance of corners)
                    movement = np.linalg.norm(corners - last_corners, axis=2).mean()
                    
                    if movement < MOVEMENT_THRESHOLD:
                        current_time = time.time()
                        
                        if stable_start_time is None:
                            stable_start_time = current_time
                            # Reset best frame for this new window
                            best_frame_data = None
                            min_error_in_window = 9999.0

                        # We are stable. Check if this is the best frame so far.
                        error = calculate_homography_error(objp_2d, corners)
                        
                        if error < min_error_in_window:
                            min_error_in_window = error
                            best_frame_data = (frame.copy(), corners.copy(), error)
                            
                        elapsed = current_time - stable_start_time
                        countdown = max(0, STATIONARY_TIME - elapsed)
                        
                        # Visual feedback for timer
                        bar_width = int((elapsed / STATIONARY_TIME) * 200)
                        cv2.rectangle(display_frame, (30, 90), (30 + bar_width, 105), (0, 255, 0), -1)
                        cv2.rectangle(display_frame, (30, 90), (230, 105), (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Stable: {countdown:.1f}s", (240, 105), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Show current best error
                        cv2.putText(display_frame, f"Best Err: {min_error_in_window:.3f}", (30, 130), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        if elapsed >= STATIONARY_TIME:
                            should_save = True
                            stable_start_time = None # Reset timer after save
                    else:
                        # Too much movement, reset timer
                        stable_start_time = None
                        best_frame_data = None
                        cv2.putText(display_frame, "Moving...", (30, 105), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    stable_start_time = None # First frame with corners
                
                last_corners = corners
            else:
                last_corners = None
                stable_start_time = None
            
            if not should_save:
                cv2.putText(display_frame, "Pattern Detected! Press 's' to save", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Use the best frame if available, otherwise current (fallback)
                if best_frame_data is not None:
                    print(f"Auto-Capture Triggered! Using BEST frame with error {best_frame_data[2]:.4f}")
                    # Note: We don't display the best frame here, we just save its data. 
                    # The display will continue showing the live feed.
                    saved_corners = best_frame_data[1]
                else:
                    saved_corners = corners # Fallback

        # OSD for Auto Mode status
        mode_color = (0, 255, 0) if auto_mode else (0, 0, 255)
        mode_text = "AUTO: ON" if auto_mode else "AUTO: OFF (Press 'a')"
        cv2.putText(display_frame, mode_text, (w - 250, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        cv2.putText(display_frame, f"Saved Frames: {count}/{MIN_FRAMES}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Calibration Feed (Max)', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Toggle Auto Mode
        if key == ord('a'):
            auto_mode = not auto_mode
            stable_start_time = None
            print(f"Auto Mode: {'ON' if auto_mode else 'OFF'}")

        # Press 's' to save the points (or auto trigger)
        if (key == ord('s') or should_save) and ret_corners:
            # If manual save, we use current corners (refined)
            # If auto save, we already set saved_corners to best_corners
            if not should_save: # Manual
                saved_corners = corners

            if saved_corners is not None:
                objpoints.append(objp)
                imgpoints.append(saved_corners)
                count += 1
                print(f"Frame {count} captured.")
                
                # Reset timer to prevent double frames immediately
                stable_start_time = None
                best_frame_data = None
                
                # Flash effect
                cv2.rectangle(display_frame, (0,0), (w,h), (255,255,255), -1)
                cv2.imshow('Calibration Feed (Max)', display_frame)
                cv2.waitKey(50)
        
        # Press 'q' to quit and calculate
        elif key == ord('q'):
            if count < MIN_FRAMES:
                print(f"Not enough frames! Need {MIN_FRAMES - count} more.")
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

    if count >= MIN_FRAMES:
        print("Calculating calibration matrix... this may take a moment.")
        h, w = gray.shape[:2]
        
        # The main calibration function
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

        calibration_data = {
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.tolist(),
            "width": w,
            "height": h,
            "reprojection_error": ret
        }

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print(f"Success! Calibration data saved to '{OUTPUT_FILE}' with error {ret:.4f}")
    else:
        print("Calibration aborted.")

if __name__ == "__main__":
    calibrate_lens()
