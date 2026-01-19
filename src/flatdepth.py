import cv2
import numpy as np
import sys
import json
import time

# --- CONFIGURATION ---
CAM_ID_LEFT = 1
CAM_ID_RIGHT = 2
DEFAULT_BASELINE = 13.2  # Fallback if no argument or file not found
FLATTEN_FILE = "flatten.json"

# StereoSGBM Tuning (Default)
MIN_DISPARITY = 0
NUM_DISPARITIES = 16*3   # Initial default
BLOCK_SIZE = 5           # Initial default (Lower usually captures finer details but noisier)
# ---------------------

def update_params(val):
    pass # Callback for trackbars (values read in loop)

def load_flatten_params():
    try:
        with open(FLATTEN_FILE, 'r') as f:
            data = json.load(f)
            mtx = np.array(data["camera_matrix"])
            dist = np.array(data["dist_coeffs"])
            return mtx, dist
    except FileNotFoundError:
        print(f"Error: '{FLATTEN_FILE}' not found. Run 'flatten_calib.py' first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading '{FLATTEN_FILE}': {e}")
        sys.exit(1)

def load_calibration(baseline_cm):
    """ Loads the calibration data for a specific baseline """
    filename = f"flattenstereo-{baseline_cm}.json"
    meta_filename = "stereo.json"
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Try to get pixel error from the calibration data itself first
        pixel_error = data.get("pixel_error", None)

        if pixel_error is None:
            # Fallback: Try to read from stereo.json metadata
            pixel_error = "N/A"
            try:
                with open(meta_filename, 'r') as f:
                    meta = json.load(f)
                    # Check if metadata matches the requested file
                    if meta.get("file") == filename or meta.get("file") == filename.split('/')[-1]:
                        pixel_error = meta.get("pixel_error", "N/A")
            except FileNotFoundError:
                pass
        
        return data, pixel_error
    except FileNotFoundError:
        print(f"Error: Calibration file '{filename}' not found.")
        print("Please run 'flatcalibrate.py' first.")
        sys.exit(1)

def main():
    # 1. Handle Arguments
    if len(sys.argv) > 1:
        try:
            baseline = float(sys.argv[1])
        except ValueError:
            print("Error: Baseline must be a number.")
            sys.exit(1)
    else:
        # Try to read the last used baseline from stereo.json
        try:
            with open("stereo.json", 'r') as f:
                meta = json.load(f)
                baseline = meta.get("user_baseline_arg", DEFAULT_BASELINE)
                print(f"No argument provided. Using last calibrated baseline: {baseline} cm")
        except:
            baseline = DEFAULT_BASELINE
            print(f"No argument or history. Using default baseline: {baseline} cm")

    # 2. Load Flatten Parameters
    flat_mtx, flat_dist = load_flatten_params()
    print(f"Loaded {FLATTEN_FILE} for pre-undistortion.")

    # 3. Load Stereo Calibration Data
    calib_data, pixel_error = load_calibration(baseline)
    
    # Convert lists back to numpy arrays
    M1 = np.array(calib_data["M1"])
    d1 = np.array(calib_data["d1"])
    M2 = np.array(calib_data["M2"])
    d2 = np.array(calib_data["d2"])
    R1 = np.array(calib_data["R1"])
    P1 = np.array(calib_data["P1"])
    R2 = np.array(calib_data["R2"])
    P2 = np.array(calib_data["P2"])
    Q = np.array(calib_data["Q"])
    
    width = int(calib_data["width"])
    height = int(calib_data["height"])

    # 4. Init Cameras
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("Error: Cameras not found.")
        sys.exit()

    # 5. Create Rectification Maps (Computed once for efficiency)
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)

    # 6. Setup StereoSGBM Matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 7. WLS Filter (Accuracy Improvement)
    wls_filter = None
    right_matcher = None
    try:
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        print("WLS Filter: ENABLED (High Accuracy Mode)")
    except AttributeError:
        print("WLS Filter: DISABLED (cv2.ximgproc not found). Install opencv-contrib-python for better results.")

    # 8. Create Trackbars
    cv2.namedWindow("Flat Stereo Depth")
    cv2.createTrackbar("Num Disparities (16x)", "Flat Stereo Depth", 6, 16, update_params) # *16
    cv2.createTrackbar("Block Size", "Flat Stereo Depth", 5, 50, update_params) 
    cv2.createTrackbar("Uniqueness", "Flat Stereo Depth", 10, 20, update_params)
    cv2.createTrackbar("Speckle Window", "Flat Stereo Depth", 100, 200, update_params)
    cv2.createTrackbar("Speckle Range", "Flat Stereo Depth", 32, 50, update_params)
    if wls_filter:
        cv2.createTrackbar("WLS Lambda (x1000)", "Flat Stereo Depth", 8, 20, update_params)
        cv2.createTrackbar("WLS Sigma (x10)", "Flat Stereo Depth", 15, 30, update_params)

    print(f"--- F L A T   Detailed Depth Map Generator ---")
    print(f"User Baseline: {baseline} cm")
    calc_baseline = calib_data.get("calculated_baseline", "N/A")
    if calc_baseline != "N/A": 
        calc_baseline = f"{calc_baseline:.2f}"
        
    print(f"Calc Baseline: {calc_baseline} cm")
    print(f"Pixel Error (RMS): {pixel_error}")
    print("Press 'Q' to quit.")

    while True:
        ret_l, raw_frame_l = cap_l.read()
        ret_r, raw_frame_r = cap_r.read()

        if not ret_l or not ret_r: break

        # A. PRE-UNDISTORT (FLAT)
        # Using the single camera calibration to flatten the lens distortion first
        frame_l = cv2.undistort(raw_frame_l, flat_mtx, flat_dist, None, flat_mtx)
        frame_r = cv2.undistort(raw_frame_r, flat_mtx, flat_dist, None, flat_mtx)

        # B. Rectify Images
        # This aligns the images so that epipolar lines are horizontal
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        
        # --- Update Stereo Parameters from Trackbars ---
        num_disp = cv2.getTrackbarPos("Num Disparities (16x)", "Flat Stereo Depth") * 16
        if num_disp < 16: num_disp = 16
        
        blk = cv2.getTrackbarPos("Block Size", "Flat Stereo Depth")
        if blk % 2 == 0: blk += 1 # Must be odd
        if blk < 5: blk = 5
        
        uniqueness = cv2.getTrackbarPos("Uniqueness", "Flat Stereo Depth")
        speckle_win = cv2.getTrackbarPos("Speckle Window", "Flat Stereo Depth")
        speckle_range = cv2.getTrackbarPos("Speckle Range", "Flat Stereo Depth")

        stereo.setNumDisparities(num_disp)
        stereo.setBlockSize(blk)
        stereo.setUniquenessRatio(uniqueness)
        stereo.setSpeckleWindowSize(speckle_win)
        stereo.setSpeckleRange(speckle_range)

        if wls_filter:
            # Update WLS params
            lam = cv2.getTrackbarPos("WLS Lambda (x1000)", "Flat Stereo Depth") * 1000.0
            sig = cv2.getTrackbarPos("WLS Sigma (x10)", "Flat Stereo Depth") / 10.0
            wls_filter.setLambda(lam)
            wls_filter.setSigmaColor(sig)
            
            # WLS needs the right matcher to match the left matcher's params
            right_matcher.setNumDisparities(num_disp)
            right_matcher.setBlockSize(blk)
            right_matcher.setUniquenessRatio(uniqueness)
            right_matcher.setSpeckleWindowSize(speckle_win)
            right_matcher.setSpeckleRange(speckle_range)

        # C. Compute Disparity
        # Convert to grayscale for matching
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        if wls_filter:
            # High Accuracy Mode: Compute both Left and Right disparity maps
            disp_l = stereo.compute(gray_l, gray_r)
            disp_r = right_matcher.compute(gray_r, gray_l)
            
            # Apply WLS Filter
            disp_filtered = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r, right_view=gray_r)
            
            disparity = disp_filtered.astype(np.float32) / 16.0
        else:
            # Standard Mode
            disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # D. Visualization
        # Normalize disparity for display (0-255)
        disp_vis = (disparity - MIN_DISPARITY) / num_disp
        disp_vis = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # E. Add Overlay Text
        info_text = f"Baseline: {baseline}cm | Flat Mode"
        cv2.putText(disp_color, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show Side-by-Side (Rectified Input vs Depth)
        combined = np.hstack((rect_l, disp_color))
        
        cv2.imshow("Flat Stereo Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
