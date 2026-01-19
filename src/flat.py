import cv2
import numpy as np
import sys
import json
import os

INPUT_FILE = "flatten.json"
CAMERA_ID = 0 if len(sys.argv) < 1 else int(sys.argv[1])
print(sys.argv)

def run_flattened_camera():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run calibration first.")
        return

    # 1. Load Calibration Data
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    w, h = data["width"], data["height"]

    # 2. Setup Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 3. Optimizing the Camera Matrix
    # alpha=0: Crop the image (zoom in) to remove black borders
    # alpha=1: Keep all pixels (black borders will be visible)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Pre-compute the undistortion map (faster than undistorting every frame individually)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    x, y, w_roi, h_roi = roi

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Option A: Simple Undistort (Slower, easiest to code)
        # flattened = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # Option B: Remapping (Faster for video)
        flattened = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # Optional: Crop the image based on ROI if alpha was used
        # flattened = flattened[y:y+h_roi, x:x+w_roi]

        # Show side-by-side comparison (resize for display if needed)
        combined = np.hstack((cv2.resize(frame, (640, 480)), cv2.resize(flattened, (640, 480))))
        
        cv2.imshow('Original vs Flattened', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_flattened_camera()