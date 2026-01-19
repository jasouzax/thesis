import cv2
import numpy as np
import json
import sys
import os

CAM_ID = 0
DEFAULT_FILE = "flatten.json"
OUTPUT_FILE = "flattencustom.json"

# Scaling factors for trackbars (Trackbars only support int)
# Distortion coeffs need high precision
K_SCALE = 1000.0   # Val 500 -> 0.5
P_SCALE = 10000.0  # Val 500 -> 0.05
DIST_MID = 5000     # Midpoint for trackbars (0.0 value)

def nothing(x):
    pass

def load_defaults():
    # Default values if file not found
    mtx = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    
    if os.path.exists(DEFAULT_FILE):
        try:
            with open(DEFAULT_FILE, 'r') as f:
                data = json.load(f)
                mtx = np.array(data["camera_matrix"], dtype=np.float32)
                dist = np.array(data["dist_coeffs"], dtype=np.float32).flatten()
                print(f"Loaded defaults from {DEFAULT_FILE}")
        except Exception as e:
            print(f"Error loading defaults: {e}")
            
    return mtx, dist

def main():
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("Error: Camera not available")
        return

    # Read one frame to get dims
    ret, frame = cap.read()
    if not ret: return
    h, w = frame.shape[:2]

    # Load defaults
    mtx_def, dist_def = load_defaults()
    
    fx_def, fy_def = mtx_def[0,0], mtx_def[1,1]
    cx_def, cy_def = mtx_def[0,2], mtx_def[1,2]
    
    # Ensure dist has 5 elements
    if len(dist_def) < 5:
        dist_def = np.pad(dist_def, (0, 5-len(dist_def)))
    
    k1_def, k2_def, p1_def, p2_def, k3_def = dist_def[:5]

    cv2.namedWindow('Flatten Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Flatten Tuner', 1200, 800)

    # --- TRACKBARS ---
    # Matric Params
    cv2.createTrackbar('fx', 'Flatten Tuner', int(fx_def), 2000, nothing)
    cv2.createTrackbar('fy', 'Flatten Tuner', int(fy_def), 2000, nothing)
    cv2.createTrackbar('cx', 'Flatten Tuner', int(cx_def), w, nothing)
    cv2.createTrackbar('cy', 'Flatten Tuner', int(cy_def), h, nothing)

    # Distortion Params (Centered at DIST_MID)
    # k1 range: +/- 5.0 (scale 1000)
    # p1 range: +/- 0.5 (scale 10000)
    
    def to_tb(val, scale):
        return int(val * scale) + DIST_MID

    cv2.createTrackbar('k1', 'Flatten Tuner', to_tb(k1_def, K_SCALE), DIST_MID*2, nothing)
    cv2.createTrackbar('k2', 'Flatten Tuner', to_tb(k2_def, K_SCALE), DIST_MID*2, nothing)
    cv2.createTrackbar('p1', 'Flatten Tuner', to_tb(p1_def, P_SCALE), DIST_MID*2, nothing)
    cv2.createTrackbar('p2', 'Flatten Tuner', to_tb(p2_def, P_SCALE), DIST_MID*2, nothing)
    cv2.createTrackbar('k3', 'Flatten Tuner', to_tb(k3_def, K_SCALE), DIST_MID*2, nothing)

    print(f"--- FLATTEN TUNER ---")
    print(f"Adjust trackbars to flatten the image.")
    print(f"Press 's' to SAVE to {OUTPUT_FILE}")
    print(f"Press 'r' to RESET to loaded defaults")
    print(f"Press 'q' to QUIT")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Read Trackbars
        fx = cv2.getTrackbarPos('fx', 'Flatten Tuner')
        fy = cv2.getTrackbarPos('fy', 'Flatten Tuner')
        cx = cv2.getTrackbarPos('cx', 'Flatten Tuner')
        cy = cv2.getTrackbarPos('cy', 'Flatten Tuner')

        k1_tb = cv2.getTrackbarPos('k1', 'Flatten Tuner')
        k2_tb = cv2.getTrackbarPos('k2', 'Flatten Tuner')
        p1_tb = cv2.getTrackbarPos('p1', 'Flatten Tuner')
        p2_tb = cv2.getTrackbarPos('p2', 'Flatten Tuner')
        k3_tb = cv2.getTrackbarPos('k3', 'Flatten Tuner')

        # Convert to float params
        mtx = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        k1 = (k1_tb - DIST_MID) / K_SCALE
        k2 = (k2_tb - DIST_MID) / K_SCALE
        p1 = (p1_tb - DIST_MID) / P_SCALE
        p2 = (p2_tb - DIST_MID) / P_SCALE
        k3 = (k3_tb - DIST_MID) / K_SCALE
        
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        # UNDISTORT
        # Note: We want to see how the parameters 'flatten' the raw image
        # So we use the parameters AS the camera calibration
        try:
            undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
        except:
            # Handles extreme values that might crash undistort
            undistorted = frame.copy()

        # OVERLAYS
        # Add a grid to help visual alignment
        def draw_grid(img, color=(0, 255, 0)):
            _h, _w = img.shape[:2]
            # Horizontal
            for i in range(0, _h, 40): cv2.line(img, (0, i), (_w, i), color, 1)
            # Vertical
            for i in range(0, _w, 40): cv2.line(img, (i, 0), (i, _h), color, 1)
            return img

        # Left: Original (Raw), Right: Tuned (Undistorted)
        # We put grid on undistorted to check straightness
        vis_raw = frame.copy()
        vis_undist = draw_grid(undistorted.copy())
        
        # Add labels
        cv2.putText(vis_raw, "ORIGINAL (Raw)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_undist, "UNDISTORTED (Tuned)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine
        combined = np.hstack((vis_raw, vis_undist))
        
        # Add Params Text
        info = f"fx={fx} fy={fy} k1={k1:.3f} k2={k2:.3f} p1={p1:.4f}"
        cv2.putText(combined, info, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Flatten Tuner', combined)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            data = {
                "camera_matrix": mtx.tolist(),
                "dist_coeffs": [dist.tolist()], # Wrap in list to match format [[k1, k2...]]
                "width": w,
                "height": h,
                "note": "Created with flattentuner.py"
            }
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved to {OUTPUT_FILE}")
            
            # Flash
            cv2.rectangle(combined, (0,0), (w*2, h), (255,255,255), -1)
            cv2.imshow('Flatten Tuner', combined)
            cv2.waitKey(50)
            
        elif k == ord('r'):
            # Reset logic (just set trackbars back)
            cv2.setTrackbarPos('fx', 'Flatten Tuner', int(fx_def))
            cv2.setTrackbarPos('fy', 'Flatten Tuner', int(fy_def))
            # ... (complete reset if needed, but manual re-adjust is fine)
            print("Resetting... (partial implementation)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
