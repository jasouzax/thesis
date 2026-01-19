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

# --- ENHANCEMENT CONFIG ---
ENABLE_CLAHE = True
ENABLE_SHARPEN = False
CONTRAST_ALPHA = 1.5     # Simple contrast control (1.0-3.0)
BRIGHTNESS_BETA = 0      # Simple brightness control (0-100)

def enhance_image(image):
    """
    Applies aggressive contrast enhancement to make checkerboard pattern pop.
    """
    # 1. Convert to Grayscale (if not already)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if ENABLE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    # 3. Simple Contrast/Brightness Boost
    # new_img = alpha * old_img + beta
    gray = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)
    
    # 4. Sharpen (Optional, can introduce noise)
    if ENABLE_SHARPEN:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)

    return gray

def calculate_homography_error(objp_2d, corners):
    """
    Calculates the reprojection error assuming a planar homography.
    """
    if corners is None or len(corners) != len(objp_2d):
        return 9999.0
    
    corners_2d = corners.reshape(-1, 2)
    
    H, mask = cv2.findHomography(objp_2d, corners_2d, cv2.RANSAC, 5.0)
    if H is None: return 9999.0
        
    projected_corners = cv2.perspectiveTransform(objp_2d.reshape(-1, 1, 2), H).reshape(-1, 2)
    error = np.linalg.norm(corners_2d - projected_corners, axis=1).mean()
    return error

def calibrate_lens():
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    objp_2d = objp[:, :2]

    objpoints = [] 
    imgpoints = [] 

    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    print(f"--- FLATTEN MAX CLEAR ---")
    print(f"ENHANCEMENTS: CLAHE={ENABLE_CLAHE}, CONTRAST={CONTRAST_ALPHA}")
    print(f"1. Show the checkerboard to the camera (Look at 'Enhanced View').")
    print(f"2. Press 's' to save a frame when corners are detected.")
    print(f"3. Press 'a' to toggle AUTO MODE (finds BEST frame in 5s window).")
    print(f"4. Need at least {MIN_FRAMES} frames. Press 'q' to finish.")

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
        
        # --- ENHANCE ---
        # We use the enhanced grayscale image for detection
        gray_enhanced = enhance_image(frame)
        
        # Find corners on the ENHANCED image
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret_corners, corners_raw = cv2.findChessboardCorners(gray_enhanced, CHECKERBOARD, flags)

        display_frame = frame.copy()
        
        if ret_corners:
            # Subpixel refine uses the ENHANCED image gradients
            corners = cv2.cornerSubPix(gray_enhanced, corners_raw, (11, 11), (-1, -1), 
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # Draw on original frame (for visual confirmation it matches reality)
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)
            
            # Draw on enhanced frame (to see what computer sees)
            vis_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_enhanced, CHECKERBOARD, corners, ret_corners)
            
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
                            # Save ORIGINAL frame, but ENHANCED corners? 
                            # Ideally we calibrate on the original image space. 
                            # But since detection was on enhanced, the corners are valid for that image.
                            # Geometry shouldn't change with contrast enhancement, so corners are valid spatial coords.
                            best_frame_data = (frame.copy(), corners.copy(), error)
                            
                        elapsed = current_time - stable_start_time
                        countdown = max(0, STATIONARY_TIME - elapsed)
                        
                        bar_width = int((elapsed / STATIONARY_TIME) * 200)
                        cv2.rectangle(display_frame, (30, 90), (30 + bar_width, 105), (0, 255, 0), -1)
                        cv2.rectangle(display_frame, (30, 90), (230, 105), (255, 255, 255), 2) # border
                        cv2.putText(display_frame, f"Stable: {countdown:.1f}s", (240, 105), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        cv2.putText(display_frame, f"Best Err: {min_error_in_window:.3f}", (30, 130), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        if elapsed >= STATIONARY_TIME:
                            should_save = True
                            stable_start_time = None
                    else:
                        stable_start_time = None
                        best_frame_data = None
                        cv2.putText(display_frame, "Moving...", (30, 105), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    stable_start_time = None
                
                last_corners = corners
            else:
                last_corners = None
                stable_start_time = None
            
            if not should_save:
                cv2.putText(display_frame, "Pattern Detected!", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if best_frame_data is not None:
                    print(f"Auto-Capture: Using frame with error {best_frame_data[2]:.4f}")
                    saved_corners = best_frame_data[1]
                else:
                    saved_corners = corners
        else:
            # No corners found
            vis_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)

        # OSD
        mode_color = (0, 255, 0) if auto_mode else (0, 0, 255)
        mode_text = "AUTO: ON" if auto_mode else "AUTO: OFF (Press 'a')"
        cv2.putText(display_frame, mode_text, (w - 250, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        cv2.putText(display_frame, f"Saved: {count}/{MIN_FRAMES}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # SHOW BOTH VIEWS
        # Resize if too big
        disp_small = cv2.resize(display_frame, (480, 360))
        enh_small = cv2.resize(vis_enhanced, (480, 360))
        
        combined = np.hstack((disp_small, enh_small))
        cv2.imshow('Left: Raw | Right: Enhanced', combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):
            auto_mode = not auto_mode
            stable_start_time = None
            print(f"Auto Mode: {'ON' if auto_mode else 'OFF'}")

        if (key == ord('s') or should_save) and ret_corners:
            if not should_save: # Manual
                saved_corners = corners

            if saved_corners is not None:
                objpoints.append(objp)
                imgpoints.append(saved_corners)
                count += 1
                print(f"Frame {count} captured.")
                stable_start_time = None
                best_frame_data = None
                cv2.waitKey(100) # pause briefly
        
        elif key == ord('q'):
            if count < MIN_FRAMES:
                print(f"Keep going! {count}/{MIN_FRAMES}")
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

    if count >= MIN_FRAMES:
        print("Calculating calibration matrix...")
        h, w = gray_enhanced.shape[:2] # use enhanced size (same as original)
        
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
        print(f"Success! Saved to '{OUTPUT_FILE}' with error {ret:.4f}")

if __name__ == "__main__":
    calibrate_lens()
