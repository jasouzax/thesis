#!/usr/bin/env python3
import sys                                                                  # System for CLI Arguments
import cv2                                                                  # OpenCV2 for Image Processing
import numpy as np                                                          # Numpy for Data Processing
import os                                                                   # OS for FileSystem interaction
import json                                                                 # JSON for calibration data
import matplotlib.pyplot as plt                                             # Matplotlib for plotting

if len(sys.argv) > 1 and sys.argv[1] == 'help':
    print('\x1b[1;32mThesis Program - Stereo Depth Map\x1b[0m')
    print('  \x1b[1mKey Actions on GUI\x1b[0m')
    print('    \x1b[2mq\x1b[0m Quit application')
    print('    \x1b[2mh\x1b[0m Toggle hand recognition')
    print('    \x1b[2mc\x1b[0m Start calibration mode')
    print('    \x1b[2md\x1b[0m Toggle depth map display')
    print('    \x1b[2ms\x1b[0m Save calibration data')
    print('    \x1b[2mSPACE\x1b[0m Capture calibration image (during calibration)')
    print('    \x1b[2mp\x1b[0m Show calibration performance plots')
    exit()

# Typing imports
from numpy.typing import NDArray                                            # NDArray Type
from typing import cast, NamedTuple, Optional, List, Any                    # General Python Types

# Configuration Variables
cam_id: tuple[int, int] = (1, 2)                                            # Camera ID for /dev/video{n}
scale: float = 1                                                          # Scale ratio for faster processing
baseline_cm: float = 7.5                                                    # Physical distance between cameras in cm
chessboard_size: tuple[int, int] = (9, 6)                                   # Chessboard inner corners (cols, rows)
square_size_mm: float = 20.0                                                # Chessboard square size in mm

# Calibration file paths
calib_file: str = 'stereo_calibration.json'
calib_results_file: str = 'calibration_results.json'

# Class Types for hand tracking (optional feature)
class HandLandmarkList:
    landmark: List[Any]
class Classification:
    label: str
    score: float
class ClassificationList:
    classification: List[Classification]
class HandsResults(NamedTuple):
    multi_hand_landmarks: Optional[List[HandLandmarkList]]
    multi_handedness: Optional[List[ClassificationList]]
    multi_hand_world_landmarks: Optional[List[HandLandmarkList]]

# Helper Functions
def error(msg: str, end: bool = True) -> None:
    print(f"\x1b[1;31mError:\x1b[0m {msg}")
    if end:
        exit()

def info(msg: str) -> None:
    print(f"\x1b[1;34mInfo:\x1b[0m {msg}")

def success(msg: str) -> None:
    print(f"\x1b[1;32mSuccess:\x1b[0m {msg}")

def load_calibration() -> tuple[bool, Optional[dict]]:
    """Load stereo calibration data from file"""
    if not os.path.exists(calib_file):
        return False, None
    
    try:
        with open(calib_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        for key in ['camera_matrix_left', 'dist_coeffs_left', 'camera_matrix_right', 
                    'dist_coeffs_right', 'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q']:
            if key in data:
                data[key] = np.array(data[key])
        
        info(f"Loaded calibration with baseline: {data.get('baseline_cm', 'unknown')}cm")
        return True, data
    except Exception as e:
        error(f"Failed to load calibration: {str(e)}", end=False)
        return False, None

def save_calibration(calib_data: dict) -> None:
    """Save stereo calibration data to file"""
    try:
        # Convert numpy arrays to lists for JSON serialization
        save_data = {}
        for key, value in calib_data.items():
            if isinstance(value, np.ndarray):
                save_data[key] = value.tolist()
            else:
                save_data[key] = value
        
        with open(calib_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        success(f"Calibration saved to {calib_file}")
    except Exception as e:
        error(f"Failed to save calibration: {str(e)}", end=False)

def save_calibration_result(baseline: float, reproj_error: float, valid_points: int) -> None:
    """Save calibration test results for analysis"""
    result = {
        'baseline_cm': baseline,
        'reprojection_error': reproj_error,
        'valid_points': valid_points,
        'depth_accuracy_score': 100.0 / (1.0 + reproj_error)
    }
    
    # Load existing results
    results = []
    if os.path.exists(calib_results_file):
        try:
            with open(calib_results_file, 'r') as f:
                results = json.load(f)
        except:
            pass
    
    # Append new result
    results.append(result)
    
    # Save updated results
    try:
        with open(calib_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        success(f"Saved calibration result for {baseline}cm baseline")
    except Exception as e:
        error(f"Failed to save result: {str(e)}", end=False)

def plot_calibration_results() -> None:
    """Generate plots comparing different baseline calibrations"""
    if not os.path.exists(calib_results_file):
        info("No calibration results to plot yet")
        return
    
    try:
        with open(calib_results_file, 'r') as f:
            results = json.load(f)
        
        if len(results) == 0:
            info("No calibration results to plot yet")
            return
        
        baselines = [r['baseline_cm'] for r in results]
        errors = [r['reprojection_error'] for r in results]
        accuracies = [r['depth_accuracy_score'] for r in results]
        points = [r['valid_points'] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Stereo Calibration Performance Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: Reprojection Error vs Baseline
        axes[0, 0].plot(baselines, errors, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Baseline Distance (cm)', fontsize=10)
        axes[0, 0].set_ylabel('Reprojection Error (px)', fontsize=10)
        axes[0, 0].set_title('Calibration Error vs Baseline', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Depth Accuracy vs Baseline
        axes[0, 1].plot(baselines, accuracies, 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Baseline Distance (cm)', fontsize=10)
        axes[0, 1].set_ylabel('Depth Accuracy Score', fontsize=10)
        axes[0, 1].set_title('Depth Accuracy vs Baseline', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Valid Points vs Baseline
        axes[1, 0].bar(range(len(baselines)), points, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Test Number', fontsize=10)
        axes[1, 0].set_ylabel('Valid Calibration Points', fontsize=10)
        axes[1, 0].set_title('Calibration Points per Test', fontweight='bold')
        axes[1, 0].set_xticks(range(len(baselines)))
        axes[1, 0].set_xticklabels([f'{b:.1f}cm' for b in baselines], rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary comparison
        axes[1, 1].axis('off')
        best_idx = np.argmax(accuracies)
        summary_text = f"Best Configuration:\n\n"
        summary_text += f"Baseline: {baselines[best_idx]:.1f} cm\n"
        summary_text += f"Error: {errors[best_idx]:.4f} px\n"
        summary_text += f"Accuracy: {accuracies[best_idx]:.2f}\n"
        summary_text += f"Points: {points[best_idx]}\n\n"
        summary_text += f"Total Tests: {len(results)}"
        
        axes[1, 1].text(0.5, 0.5, summary_text, 
                       transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('calibration_analysis.png', dpi=300, bbox_inches='tight')
        success("Calibration plots saved to calibration_analysis.png")
        plt.show()
        
    except Exception as e:
        error(f"Failed to plot results: {str(e)}", end=False)

def run_stereo_calibration(frames_left: List[NDArray], frames_right: List[NDArray], 
                          img_shape: tuple[int, int]) -> Optional[dict]:
    """Perform stereo calibration from captured frame pairs"""
    if len(frames_left) < 5:
        error("Need at least 5 calibration images", end=False)
        return None
    
    info("Processing calibration images...")
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= (square_size_mm / 1000.0)  # Convert to meters
    
    obj_points: List[NDArray] = []
    img_points_left: List[NDArray] = []
    img_points_right: List[NDArray] = []
    
    # Find chessboard corners in all frames
    for i, (frame_l, frame_r) in enumerate(zip(frames_left, frames_right)):
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)
        
        if ret_l and ret_r:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            
            obj_points.append(objp)
            img_points_left.append(corners_l)
            img_points_right.append(corners_r)
            info(f"Frame {i+1}/{len(frames_left)}: Valid")
        else:
            info(f"Frame {i+1}/{len(frames_left)}: Skipped (no pattern)")
    
    if len(obj_points) < 5:
        error("Not enough valid calibration images", end=False)
        return None
    
    info(f"Calibrating with {len(obj_points)} valid image pairs...")
    
    # Calibrate individual cameras
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        obj_points, img_points_left, img_shape, None, None)
    
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        obj_points, img_points_right, img_shape, None, None)
    
    info(f"Individual calibration - Left: {ret_l:.4f}, Right: {ret_r:.4f}")
    
    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        mtx_l, dist_l, mtx_r, dist_r,
        img_shape, criteria=criteria, flags=flags)
    
    info(f"Stereo calibration reprojection error: {ret:.4f} pixels")
    
    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r,
        img_shape, R, T, alpha=0)
    
    # Compute rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, img_shape, cv2.CV_16SC2)
    
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, img_shape, cv2.CV_16SC2)
    
    calculated_baseline = np.linalg.norm(T) * 100  # Convert to cm
    info(f"Calculated baseline from calibration: {calculated_baseline:.2f} cm")
    
    calib_data = {
        'camera_matrix_left': mtx_l,
        'dist_coeffs_left': dist_l,
        'camera_matrix_right': mtx_r,
        'dist_coeffs_right': dist_r,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'map1_left': map1_left,
        'map2_left': map2_left,
        'map1_right': map1_right,
        'map2_right': map2_right,
        'reprojection_error': ret,
        'baseline_cm': baseline_cm,
        'calculated_baseline_cm': calculated_baseline,
        'valid_points': len(obj_points[0]) * len(obj_points),
        'image_shape': img_shape
    }
    
    success(f"Stereo calibration complete! Error: {ret:.4f}px")
    
    # Save result for analysis
    save_calibration_result(baseline_cm, ret, len(obj_points[0]) * len(obj_points))
    
    return calib_data

def compute_depth_map(frame_left: NDArray, frame_right: NDArray, 
                     calib_data: dict, show_colored: bool = True) -> NDArray:
    """Compute depth map from rectified stereo pair"""
    # Rectify images
    rect_left = cv2.remap(frame_left, calib_data['map1_left'], 
                         calib_data['map2_left'], cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, calib_data['map1_right'], 
                          calib_data['map2_right'], cv2.INTER_LINEAR)
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    
    # Create stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=16*10, blockSize=15)
    
    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    if show_colored:
        # Normalize and colorize
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(disparity_normalized.astype(np.uint8), 
                                         cv2.COLORMAP_JET)
        return depth_colored
    
    return disparity

# State variables
hand_gesture: bool = False
calibration_mode: bool = False
show_depth: bool = False
calibration_frames_left: List[NDArray] = []
calibration_frames_right: List[NDArray] = []
calibration_data: Optional[dict] = None

# Try to load existing calibration
has_calibration, calibration_data = load_calibration()

if not has_calibration:
    info("No calibration found. Press 'c' to start calibration mode")
    info(f"Current baseline setting: {baseline_cm} cm")
    info("To test different baselines, adjust cameras and recalibrate")

# Initialize hand tracking (optional - only if mediapipe available)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_tracking_available = True
except ImportError:
    hand_tracking_available = False
    info("Mediapipe not available - hand tracking disabled")

# Initialize the two cameras
print('Connecting to cameras')
cam_left, cam_right = [cv2.VideoCapture(n) for n in cam_id]
if err := (not cam_left.isOpened()) + ((not cam_right.isOpened()) << 1):
    error(f"Could not open {['', 'left', 'right', 'both'][err]} camera/s")

# Eliminate buffer for most latest frame capture
cam_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Get camera dimensions
print('Getting camera dimensions')
ret, frame = cast(tuple[bool, NDArray[np.uint8]], cam_left.read())
if not ret:
    error('Failed to get dimensions from camera')
height, width = frame.shape[:2]
height = int(height * scale)
width = int(width * scale)

print(f"Processing at resolution: {width}x{height}")

# Create windows
print('Starting graphical window')
cv2.namedWindow('Thesis', cv2.WINDOW_NORMAL)

info("Application ready - press 'h' for help (or run with 'help' argument)")

while True:
    # Read frames from both cameras simultaneously
    cam_left.grab()
    cam_right.grab()
    ret_left, frame_left = cam_left.retrieve()
    ret_right, frame_right = cam_right.retrieve()
    
    if err := (not ret_left) + ((not ret_right) << 1):
        error(f"Could not capture video from {['', 'left', 'right', 'both'][err]} camera/s")

    # Resize frames
    frame_left = cv2.resize(frame_left, (width, height))
    frame_right = cv2.resize(frame_right, (width, height))
    
    # Flip frames (camera is upside down)
    # frame_left = cv2.flip(frame_left, -1)
    # frame_right = cv2.flip(frame_right, -1)
    
    # Display frame to use
    display_left = frame_left.copy()
    display_right = frame_right.copy()
    
    # Calibration mode - show chessboard detection
    if calibration_mode:
        gray_l = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)
        
        if ret_l:
            cv2.drawChessboardCorners(display_left, chessboard_size, corners_l, ret_l)
        if ret_r:
            cv2.drawChessboardCorners(display_right, chessboard_size, corners_r, ret_r)
        
        status = "READY" if (ret_l and ret_r) else "NO PATTERN"
        color = (0, 255, 0) if (ret_l and ret_r) else (0, 0, 255)
        
        cv2.putText(display_left, f"CALIBRATION: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_left, f"Captured: {len(calibration_frames_left)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_left, "SPACE=Capture, ESC=Process", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Depth map mode
    elif show_depth and calibration_data is not None:
        depth_map = compute_depth_map(frame_left, frame_right, calibration_data)
        display_right = depth_map
        
        cv2.putText(display_left, "DEPTH MAP MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Hand gesture detection
    elif hand_gesture and hand_tracking_available:
        frame_rgb = cv2.cvtColor(display_right, cv2.COLOR_BGR2RGB)
        results = cast(HandsResults, hands.process(frame_rgb))
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                 results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    cast(List[tuple[int, int]], mp_hands.HAND_CONNECTIONS),
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        display_right = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Display combined view
    cv2.imshow('Thesis', np.hstack((display_left, display_right)))

    # Key handler
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q') or key == 27:
        if calibration_mode:
            info("Exiting calibration mode")
            calibration_mode = False
            calibration_frames_left.clear()
            calibration_frames_right.clear()
        else:
            break
    
    # Toggle hand detection
    elif key == ord('h') and hand_tracking_available:
        hand_gesture = not hand_gesture
        show_depth = False
        print(f"Hand gesture detection: {'ON' if hand_gesture else 'OFF'}")
    
    # Start calibration
    elif key == ord('c'):
        if not calibration_mode:
            calibration_mode = True
            calibration_frames_left.clear()
            calibration_frames_right.clear()
            info("Calibration mode started - show chessboard pattern")
            info(f"Chessboard: {chessboard_size[0]}x{chessboard_size[1]} inner corners")
            info(f"Square size: {square_size_mm}mm")
        else:
            # Process calibration
            info("Processing calibration data...")
            calibration_data = run_stereo_calibration(
                calibration_frames_left, calibration_frames_right, (width, height))
            
            if calibration_data is not None:
                save_calibration(calibration_data)
                has_calibration = True
            
            calibration_mode = False
            calibration_frames_left.clear()
            calibration_frames_right.clear()
    
    # Capture calibration frame
    elif key == 32 and calibration_mode:  # SPACE
        gray_l = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        ret_l, _ = cv2.findChessboardCorners(gray_l, chessboard_size, None)
        ret_r, _ = cv2.findChessboardCorners(gray_r, chessboard_size, None)
        
        if ret_l and ret_r:
            calibration_frames_left.append(frame_left.copy())
            calibration_frames_right.append(frame_right.copy())
            success(f"Captured frame {len(calibration_frames_left)}")
        else:
            info("Pattern not detected in both cameras")
    
    # Toggle depth map
    elif key == ord('d'):
        if has_calibration:
            show_depth = not show_depth
            hand_gesture = False
            print(f"Depth map display: {'ON' if show_depth else 'OFF'}")
        else:
            info("Calibrate cameras first (press 'c')")
    
    # Save current calibration
    elif key == ord('s'):
        if calibration_data is not None:
            save_calibration(calibration_data)
        else:
            info("No calibration data to save")
    
    # Show performance plots
    elif key == ord('p'):
        plot_calibration_results()

# Cleanup
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
print('\x1b[1;32mCleanup complete\x1b[0m')