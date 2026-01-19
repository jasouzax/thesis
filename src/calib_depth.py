import cv2
import numpy as np
import sys
import json
import os
import time
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- CONFIGURATION SECTION ---
CHECKERBOARD_DIMS = (9, 6)
SQUARE_SIZE_CM = 2.5
CAM_ID_LEFT = 1
CAM_ID_RIGHT = 2

# Auto-Capture Settings
AUTO_CAPTURE_DELAY = 2.0    # Seconds board must be still to trigger capture
MOVEMENT_THRESHOLD = 3.0    # Max pixel movement allowed to count as "still"
POST_CAPTURE_COOLDOWN = 2.0 # Seconds to wait after capture before scanning again
# -----------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def draw_overlays(img, mode='lines'):
    h, w = img.shape[:2]
    cx = w // 2
    # Pitch Lines (Green)
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    # Yaw Center Line (Cyan)
    cv2.line(img, (cx, 0), (cx, h), (255, 255, 0), 2)
    # Roll Grid (Gray) - Only in Grid mode
    if mode == 'grid':
        for x in range(0, w, 40):
            cv2.line(img, (x, 0), (x, h), (100, 100, 100), 1)
    return img

def calculate_movement(curr_corners, prev_corners):
    """ Returns average pixel distance between current and previous corners """
    if prev_corners is None or curr_corners is None:
        return 999.9
    # Calculate Euclidean distance for all corresponding corners
    diff = np.linalg.norm(curr_corners - prev_corners, axis=2)
    return np.mean(diff)

def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <baseline_cm>")
        sys.exit(1)
    baseline_arg = float(sys.argv[1])

    print("--- Auto-Capture Stereo Calibration ---")
    print("CONTROLS:")
    print("  [A]          Toggle Auto-Capture Mode (Hands-free)")
    print("  [O]          Toggle Overlay Mode (Check Roll)")
    print("  [ENTER]      Start/Stop Scanning")
    print("  [SPACE]      Manual Capture")
    print("  [Q]          Quit")

    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("Error opening cameras.")
        sys.exit()

    # Prep Calibration Data
    objp = np.zeros((CHECKERBOARD_DIMS[0]*CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE_CM

    objpoints = [] 
    imgpoints_l = []
    imgpoints_r = []

    # State Variables
    scanning_mode = False
    overlay_mode = False
    auto_capture_mode = False
    image_count = 0
    
    # Auto-capture logic vars
    prev_corners_l = None
    stability_start_time = None
    last_capture_time = 0
    
    # Read first frame for dimensions
    _, frame_l = cap_l.read()
    h, w = frame_l.shape[:2]

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r: break

        vis_l = frame_l.copy()
        vis_r = frame_r.copy()
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # Detect Checkerboards (Only in Scanning Mode to save FPS)
        ret_c_l, ret_c_r = False, False
        corners_l, corners_r = None, None
        
        if scanning_mode:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            ret_c_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD_DIMS, flags)
            ret_c_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD_DIMS, flags)

            # Refine and Draw
            if ret_c_l:
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(vis_l, CHECKERBOARD_DIMS, corners_l, ret_c_l)
            if ret_c_r:
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(vis_r, CHECKERBOARD_DIMS, corners_r, ret_c_r)

        # --- LOGIC & UI ---
        
        # 1. VISUALIZATION MODES
        if overlay_mode:
            blended = cv2.addWeighted(vis_l, 0.5, vis_r, 0.5, 0)
            blended = draw_overlays(blended, mode='grid')
            combined = np.hstack((blended, blended))
            mode_text = "MODE: OVERLAY"
        else:
            vis_l = draw_overlays(vis_l, mode='lines')
            vis_r = draw_overlays(vis_r, mode='lines')
            combined = np.hstack((vis_l, vis_r))
            mode_text = "MODE: SIDE-BY-SIDE"

        # 2. SCANNING & AUTO CAPTURE LOGIC
        capture_now = False
        status_text = "ALIGNMENT MODE (Press ENTER)"
        status_color = (0, 255, 255) # Yellow
        progress = 0.0

        if scanning_mode:
            current_time = time.time()
            
            if ret_c_l and ret_c_r:
                # Board Detected
                if auto_capture_mode:
                    # Check for cooldown
                    if current_time - last_capture_time < POST_CAPTURE_COOLDOWN:
                        status_text = "Cooldown..."
                        status_color = (100, 100, 100)
                        stability_start_time = None # Reset timer
                    else:
                        # Check Movement
                        movement = calculate_movement(corners_l, prev_corners_l)
                        
                        if movement < MOVEMENT_THRESHOLD:
                            # It is stable
                            if stability_start_time is None:
                                stability_start_time = current_time
                            
                            elapsed = current_time - stability_start_time
                            progress = min(elapsed / AUTO_CAPTURE_DELAY, 1.0)
                            
                            status_text = f"STABLE: {int(progress*100)}%"
                            status_color = (0, 255, 0)

                            if elapsed >= AUTO_CAPTURE_DELAY:
                                capture_now = True
                                stability_start_time = None # Reset
                        else:
                            # Moved too much
                            stability_start_time = None
                            status_text = "Movement Detected"
                            status_color = (0, 0, 255)
                        
                        prev_corners_l = corners_l
                else:
                    status_text = "READY (Press SPACE)"
                    status_color = (0, 255, 0)
            else:
                stability_start_time = None
                status_text = "Looking for board..."
                status_color = (0, 0, 255)

        # 3. DRAW UI
        # Top Status Bar
        cv2.putText(combined, f"{mode_text} | Captured: {image_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(combined, status_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # Auto Mode Indicator
        if auto_capture_mode:
            cv2.putText(combined, "[AUTO ACTIVE]", (w*2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Draw Progress Bar at bottom
            if scanning_mode and progress > 0:
                bar_width = int((w*2) * progress)
                cv2.rectangle(combined, (0, h-20), (bar_width, h), (0, 255, 0), -1)

        cv2.imshow('Stereo Calib', combined)

        # 4. CAPTURE EXECUTION
        # Triggered by Auto-Timer OR Spacebar
        k = cv2.waitKey(1) & 0xFF
        if k == 32: capture_now = True

        if capture_now and scanning_mode and ret_c_l and ret_c_r:
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            objpoints.append(objp)
            image_count += 1
            last_capture_time = time.time()
            print(f"Captured Image #{image_count}")
            
            # White Flash Effect
            cv2.rectangle(combined, (0,0), (w*2, h), (255,255,255), -1)
            cv2.imshow('Stereo Calib', combined)
            cv2.waitKey(50)

        # 5. KEYBOARD INPUTS
        if k == 27 or k == ord('q'): break
        elif k == ord('a'): 
            auto_capture_mode = not auto_capture_mode
            stability_start_time = None
            print(f"Auto-Capture: {auto_capture_mode}")
        elif k == ord('o'): overlay_mode = not overlay_mode
        elif k == 13: # Enter
            scanning_mode = not scanning_mode
            if not scanning_mode and image_count > 0:
                 break # Stop and calibrate

    # --- CALIBRATION (Same as before) ---
    cv2.destroyAllWindows()
    if image_count < 1: sys.exit()
    
    print("Calibrating...")
    img_size = (w, h)
    
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
    
    flags = cv2.CALIB_FIX_INTRINSIC
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret_s, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, img_size, criteria=crit, flags=flags
    )

    # Compute Rectification Transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, d1, M2, d2, img_size, R, T)

    # --- METRICS CALCULATION ---
    # 1. Individual RMS (Left vs Right)
    print("Calculating individual reprojection errors...")
    def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        total_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            total_points += 1
        return total_error / total_points

    # We need rvecs/tvecs from single calibration which we didn't save, so we re-estimate or just use the overall RMS from stereoCalibrate which is good enough for now,
    # OR better yet, let's use the individual errors returned by stereoCalibrate (it returns overall RMS).
    # Actually, stereoCalibrate minimizes the error for both. 
    # Let's verify individual RMS using the extrinsics.
    
    # 2. Calculated Baseline
    calc_baseline = np.linalg.norm(T)

    # 3. Horizontal FOV
    fov_x_l = 2 * np.arctan(w / (2 * M1[0, 0])) * 180 / np.pi
    fov_x_r = 2 * np.arctan(w / (2 * M2[0, 0])) * 180 / np.pi
    
    print(f"RMS Stereo: {ret_s:.4f}")
    print(f"RMS Left (Intrinsics): {ret_l:.4f}")
    print(f"RMS Right (Intrinsics): {ret_r:.4f}")
    print(f"Baseline: {calc_baseline:.4f} cm")
    print(f"FOV Left: {fov_x_l:.2f} deg, Right: {fov_x_r:.2f} deg")

    data = { 
        "timestamp": time.time(),
        "baseline": baseline_arg,
        "width": w, "height": h, 
        "M1": M1, "d1": d1, "M2": M2, "d2": d2, 
        "R": R, "T": T, "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q, 
        "pixel_error": ret_s,
        "pixel_error_left": ret_l,
        "pixel_error_right": ret_r,
        "calculated_baseline": calc_baseline,
        "fov_left": fov_x_l,
        "fov_right": fov_x_r
    }
    
    fname = f"stereo-{baseline_arg}.json"
    with open(fname, 'w') as f: json.dump(data, f, cls=NumpyEncoder, indent=4)

    # --- HISTORICAL TRACKING ---
    history = []
    if os.path.exists("stereo.json"):
        try:
            with open("stereo.json", 'r') as f:
                existing = json.load(f)
                history = existing.get("history", [])
                # Backward compatibility: if no history, maybe create it from the root if it was a valid calib? 
                # For now, just start appending.
        except:
             pass
    
    # Append current run (filter out heavy matrices for the aggregate file if desired? 
    # User said "include all data per baseline like rms intrinsics, fov left and right, etc." 
    # so we will include the summary data but maybe not the full matrices to keep stereo.json readable?
    # Actually, the user asked to "include all data per baseline". 
    # But usually stereo.json is a summary. Let's include the metrics and the pointer to the file.
    # The full matrices are in the individual file.
    # However, to be safe and strictly follow "include all data", I will include the metrics the user listed.
    
    summary_entry = {
        "timestamp": time.time(),
        "baseline": baseline_arg,
        "file": fname,
        "pixel_error": ret_s,
        "pixel_error_left": ret_l,
        "pixel_error_right": ret_r,
        "calculated_baseline": calc_baseline,
        "fov_left": fov_x_l,
        "fov_right": fov_x_r
    }
    history.append(summary_entry)

    # Save Graph
    if HAS_MATPLOTLIB and len(history) > 1:
        plot_calibration_history(history)
    elif not HAS_MATPLOTLIB:
        print("\n[WARNING] matplotlib not found. Skipping graph generation.")
        print("Install it via: pip install matplotlib")

    # Save stereo.json with HEAD pointing to current, plus history
    with open("stereo.json", 'w') as f: 
        json.dump({
            "baseline": baseline_arg, 
            "file": fname, 
            "pixel_error": ret_s,
            "calculated_baseline": calc_baseline,
            "user_baseline_arg": baseline_arg, # Added for depth.py compatibility
            "history": history
        }, f, cls=NumpyEncoder, indent=4)
        
    print(f"\nCalibration saved to {fname}")
    print(f"Summary updated in stereo.json (Total records: {len(history)})")

def plot_calibration_history(history):
    print("\nGenerating Calibration History Graph...")
    
    # Sort by baseline
    # Only include entries that have 'baseline' and 'pixel_error'
    valid_data = [h for h in history if 'baseline' in h and 'pixel_error' in h]
    valid_data.sort(key=lambda x: x['baseline'])
    
    if not valid_data: return

    baselines = [h['baseline'] for h in valid_data]
    errors = [h['pixel_error'] for h in valid_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(baselines, errors, 'o-', label='Stereo RMS Error')
    
    # Try adding individual errors if available
    if all('pixel_error_left' in h for h in valid_data):
        err_l = [h['pixel_error_left'] for h in valid_data]
        plt.plot(baselines, err_l, 'x--', alpha=0.5, label='Left RMS')
        
    if all('pixel_error_right' in h for h in valid_data):
        err_r = [h['pixel_error_right'] for h in valid_data]
        plt.plot(baselines, err_r, 'x--', alpha=0.5, label='Right RMS')

    plt.title('Stereo Calibration Error vs Baseline')
    plt.xlabel('Baseline (cm)')
    plt.ylabel('RMS Re-projection Error (pixels)')
    plt.grid(True)
    plt.legend()
    plt.savefig('calibration_graph.png')
    print("Graph saved to 'calibration_graph.png'")
    # plt.show() # Don't block execution

if __name__ == "__main__":
    main()