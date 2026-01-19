#!/usr/bin/env python3
import sys                                                                  # System for CLI Arguments
import cv2                                                                  # OpenCV2 for Image Processing
import numpy as np                                                          # Numpy for Data Processing
import mediapipe as mp                                                      # MediaPipe for Hand Recognition
from mediapipe.python.solutions import hands as mp_hands                    #   MediaPipe Hand
from mediapipe.python.solutions import drawing_utils as mp_drawing          #   MediaPipe Drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles  #   MediaPipe Drawing Styles
import os                                                                   # OS for FileSystem interaction
import torch                                                                # Torch for AI model execution

# Typing imports
from numpy.typing import NDArray                                            # NDArray Type
from typing import cast, NamedTuple, Optional, List, Any                    # General Python Types
from dataclasses import dataclass                                           # Dataclasses

# Configuration Variables
board:tuple[int,int]    = (9,6)                                             # Inner corners of checkerboard
square_size:int         = 19                                                # Size of square in mm
cam_id:tuple[int,int]   = (2,0)                                             # Camera ID for /dev/video{n}
scale:float             = 0.5                                               # Scale ratio for faster processing (reduced for wide FOV)
baseline:float          = 60.0                                              # Distance between cameras in mm

# Class Types
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
@dataclass
class StereoCalibration:
    K_left: np.ndarray
    dist_left: np.ndarray
    K_right: np.ndarray
    dist_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
@dataclass
class RectificationMaps:
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    Q: np.ndarray

# Helper Functions
def error(msg:str, end:bool=True) -> None:
    print(f"\x1b[1;31mError:\x1b[0m {msg}")
    if end: exit()

# Calibration Variables
k_min = 99999999999999999
kl_min = 99999999999999999
kr_min = 99999999999999999
go = False
objp = np.zeros((board[0]*board[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:board[0],0:board[1]].T.reshape(-1,2)
objp *= square_size
objpoints:list[NDArray[np.float32]] = []
points_left:list[NDArray[np.float32]] = []
points_right:list[NDArray[np.float32]] = []
count = 0
gray_left:NDArray[np.uint8]|None = None
gray_right:NDArray[np.uint8]|None = None

# Hand gesture recognition Variables
hand_gesture:bool = False
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Stereo Variables
ready = False
calib:None|StereoCalibration = None
rectm:None|RectificationMaps = None

# Depth
midas = None
transform = None

# Optimized parameters for wide FOV and small baseline
block_size = 7  # Increased for better matching
stereo = None

# Initialize the two cameras
print('Connecting to cameras')
cam_left, cam_right = [cv2.VideoCapture(n) for n in cam_id]
if err:=(not cam_left.isOpened())+((not cam_right.isOpened())<<1):
    error(f"Could not open {['','left','right','both'][err]} camera/s")

# Eliminate buffer for most latest frame capture
cam_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# Get camera dimensions
print('Getting camera dimensions')
ret, frame = cast(tuple[bool, NDArray[np.uint8]], cam_left.read())
if not ret:
    error('Failed to get dimensions from camera')
height, width = frame.shape[:2]
height = int(height*scale)
width = int(width*scale)

print(f"Processing at resolution: {width}x{height}")

# Create windows
print('Starting graphical window')
cv2.namedWindow('Thesis', cv2.WINDOW_NORMAL)

# WLS Filter for better disparity (optional but recommended)
wls_filter = None

while True:
    # Read frames from both cameras simultaneously
    cam_left.grab()
    cam_right.grab()
    ret_left, frame_left = cam_left.retrieve()
    ret_right, frame_right = cam_right.retrieve()
    
    if err:=(not ret_left)+((not ret_right)<<1):
        error(f"Could not capture video from {['','left','right','both'][err]} camera/s")

    # Apply consistent flipping for both calibration and display
    #frame_left = cv2.resize(frame_left, (width, height))
    #frame_right = cv2.resize(frame_right, (width, height))
    
    # Hand gesture detection in right camera
    if hand_gesture:
        frame_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        results = cast(HandsResults, hands.process(frame_rgb))
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    cast(List[tuple[int, int]], mp_hands.HAND_CONNECTIONS),
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        frame_right = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Depth Map Generation
    color_disparity = None
    if ready and rectm is not None:
        # Rectify frames
        rect_left = cv2.remap(frame_left, rectm.map_left_x, rectm.map_left_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, rectm.map_right_x, rectm.map_right_y, cv2.INTER_LINEAR)
        
        # Convert to grayscale
        gray_rect_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_rect_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity with WLS filtering if available
        if wls_filter is not None:
            disparity_left = stereo.compute(gray_rect_left, gray_rect_right)
            disparity_right = right_matcher.compute(gray_rect_right, gray_rect_left)
            disparity = wls_filter.filter(disparity_left, gray_rect_left, None, disparity_right)
            disparity = disparity.astype(np.float32) / 16.0
        else:
            disparity = stereo.compute(gray_rect_left, gray_rect_right).astype(np.float32) / 16.0
        
        # Filter invalid disparities
        disparity[disparity <= 0] = 0.1
        disparity[disparity > 96] = 0.1
        
        # Visualize disparity
        norm_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        color_disparity = cv2.applyColorMap(norm_disparity, cv2.COLORMAP_JET)
        
        # Show rectified images with epipolar lines for verification
        rect_display_left = rect_left.copy()
        rect_display_right = rect_right.copy()
        
        # Draw horizontal lines to verify rectification
        for i in range(0, height, 30):
            cv2.line(rect_display_left, (0, i), (width, i), (0, 255, 0), 1)
            cv2.line(rect_display_right, (0, i), (width, i), (0, 255, 0), 1)
        
        # Display depth map
        #cv2.imshow('Depth Map', np.hstack((rect_display_left, rect_display_right, color_disparity)))
    
    # MiDAS
    if midas is not None:
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        input_batch = transform(frame_right).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_right.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    
        # Convert to numpy array
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = depth_map_normalized.astype(np.uint8)
        
        # Apply colormap for better visualization
        frame_right = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_MAGMA)
        

    # Display original camera feeds
    frame_left = cv2.flip(frame_left, -1)
    frame_right = cv2.flip(frame_right,-1)

    cv2.imshow('Thesis', np.hstack((frame_left, frame_right) if color_disparity is None else (frame_left, frame_right, color_disparity)))

    # Key handler
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' or esc is pressed
    if key == ord('q') or key == 27:
        break
    # Toggle automatic checkerboard capture
    elif key == ord('w'):
        go = not go
        print(f"Go is {go}")
    # Points based calibration system
    elif key == ord('k') or go:
        # Use the same flipped frames for calibration
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, board, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, board, None)
        
        if err:=(not ret_left)+((not ret_right)<<1):
            pass
            #error(f"Checkerboard not found in {['','left','right','both'][err]} image/s", False)
        else:
            objpoints.append(objp)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
            
            points_left.append(corners_left)
            points_right.append(corners_right)

            try:
                # Use CALIB_RATIONAL_MODEL for wide-angle lenses
                calib_flags = (cv2.CALIB_RATIONAL_MODEL | 
                            cv2.CALIB_THIN_PRISM_MODEL |
                            cv2.CALIB_TILTED_MODEL)
                
                # Calibrate each camera individually
                ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
                    objpoints, points_left, gray_left.shape[::-1], None, None,
                    flags=calib_flags
                )
                
                ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
                    objpoints, points_right, gray_right.shape[::-1], None, None,
                    flags=calib_flags
                )
                
                print(f"Left camera RMS: {ret_left:.3f}")
                print(f"Right camera RMS: {ret_right:.3f}")
                
                # Stereo calibration
                flags = cv2.CALIB_FIX_INTRINSIC
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
                
                ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
                    objpoints, points_left, points_right,
                    K_left, dist_left, K_right, dist_right,
                    gray_left.shape[::-1], 
                    criteria=criteria,
                    flags=flags
                )

                if ret_left < kl_min or ret_right < kr_min:
                    # Save calibration data
                    np.savez('stereo_calibration.npz',
                        K_left=K_left, dist_left=dist_left,
                        K_right=K_right, dist_right=dist_right,
                        R=R, T=T, E=E, F=F
                    )
                    k_min = ret
                    kl_min = ret_left
                    kr_min = ret_right
                    print(f'\x1b[1;32mSuccess:\x1b[0m Stereo calibration complete (RMS Error: {ret:.3f})')
                else:
                    print(f'\x1b[1;31mFailure:\x1b[0m {ret} >= {k_min}')
                    objpoints.pop()
                    points_left.pop()
                    points_right.pop()
                print(f"Baseline: {np.linalg.norm(T):.1f}mm")
                
            except cv2.error as e:
                error(f'Calibration failed: {str(e)}', end=False)
                print('Try capturing more images with the checkerboard in different positions and angles')
                objpoints.pop()
                points_left.pop()
                points_right.pop()
    # Capture for calibration
    elif key == ord('c'):
        # Use the same flipped frames for calibration
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, board, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, board, None)
        
        if err:=(not ret_left)+((not ret_right)<<1):
            error(f"Checkerboard not found in {['','left','right','both'][err]} image/s", False)
        else:
            objpoints.append(objp)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
            
            points_left.append(corners_left)
            points_right.append(corners_right)
            count += 1
            print(f"\x1b[1;33mNote:\x1b[0m Captured {count} calibration image/s")
    # Generate Calibration
    elif key == ord('g'):
        print('Generating Stereo Calibration')
        
        if len(objpoints) < 10:
            error('Not enough calibration images. Capture at least 10 images.', False)
            continue
        if gray_left is None or gray_right is None:
            error('No calibration images captured', False)
            continue
        
        try:
            # Use CALIB_RATIONAL_MODEL for wide-angle lenses
            calib_flags = (cv2.CALIB_RATIONAL_MODEL | 
                          cv2.CALIB_THIN_PRISM_MODEL |
                          cv2.CALIB_TILTED_MODEL)
            
            # Calibrate each camera individually
            ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
                objpoints, points_left, gray_left.shape[::-1], None, None,
                flags=calib_flags
            )
            
            ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
                objpoints, points_right, gray_right.shape[::-1], None, None,
                flags=calib_flags
            )
            
            print(f"Left camera RMS: {ret_left:.3f}")
            print(f"Right camera RMS: {ret_right:.3f}")
            
            # Stereo calibration
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            
            ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
                objpoints, points_left, points_right,
                K_left, dist_left, K_right, dist_right,
                gray_left.shape[::-1], 
                criteria=criteria,
                flags=flags
            )
            
            # Save calibration data
            np.savez('stereo_calibration.npz',
                K_left=K_left, dist_left=dist_left,
                K_right=K_right, dist_right=dist_right,
                R=R, T=T, E=E, F=F
            )
            
            print(f'\x1b[1;32mSuccess:\x1b[0m Stereo calibration complete (RMS Error: {ret:.3f})')
            print(f"Baseline: {np.linalg.norm(T):.1f}mm")
            
        except cv2.error as e:
            error(f'Calibration failed: {str(e)}', end=False)
            print('Try capturing more images with the checkerboard in different positions and angles')
    # Compute Rectification map
    elif key == ord('r'):
        print('Generating Rectification Map')
        
        if not os.path.exists('stereo_calibration.npz'):
            error('No calibration file found. Press "g" to generate calibration first.', False)
            continue
        
        # Load calibrations
        calib_data = np.load('stereo_calibration.npz')
        K_left = calib_data['K_left']
        dist_left = calib_data['dist_left']
        K_right = calib_data['K_right']
        dist_right = calib_data['dist_right']
        R = calib_data['R']
        T = calib_data['T']

        # Compute rectification transforms (alpha=1 keeps all pixels)
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right,
            (width, height), R, T, alpha=1
        )

        # Compute rectification maps
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            K_left, dist_left, R1, P1, (width, height), cv2.CV_32FC1
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            K_right, dist_right, R2, P2, (width, height), cv2.CV_32FC1
        )

        # Save maps
        np.savez('rectification_maps.npz',
            map_left_x=map_left_x, map_left_y=map_left_y,
            map_right_x=map_right_x, map_right_y=map_right_y,
            Q=Q
        )
        
        print('\x1b[1;32mSuccess:\x1b[0m Rectification maps generated')
        print('Restart the program to use the depth map')
    # Toggle hand detection
    elif key == ord('h'):
        hand_gesture = not hand_gesture
        print(f"Hand gesture detection: {'ON' if hand_gesture else 'OFF'}")
    # Toggle stereo depth map
    elif key == ord('s'):
        if ready: ready = False
        else:
            if os.path.exists('stereo_calibration.npz') and os.path.exists('rectification_maps.npz'):
                try:
                    calib_data = np.load('stereo_calibration.npz')
                    rectm_data = np.load('rectification_maps.npz')
                    
                    calib = StereoCalibration(
                        K_left=calib_data['K_left'],
                        dist_left=calib_data['dist_left'],
                        K_right=calib_data['K_right'],
                        dist_right=calib_data['dist_right'],
                        R=calib_data['R'],
                        T=calib_data['T'],
                        E=calib_data['E'],
                        F=calib_data['F']
                    )
                    
                    rectm = RectificationMaps(
                        map_left_x=rectm_data['map_left_x'],
                        map_left_y=rectm_data['map_left_y'],
                        map_right_x=rectm_data['map_right_x'],
                        map_right_y=rectm_data['map_right_y'],
                        Q=rectm_data['Q']
                    )
                    
                    ready = True
                    
                    # Optimized SGBM parameters for 6cm baseline and wide FOV
                    stereo = cv2.StereoSGBM_create(
                        minDisparity=0,
                        numDisparities=96,  # Reduced for small baseline (must be divisible by 16)
                        blockSize=block_size,
                        P1=8 * 3 * block_size**2,
                        P2=32 * 3 * block_size**2,
                        disp12MaxDiff=1,
                        uniquenessRatio=15,  # Increased for better matching
                        speckleWindowSize=150,  # Increased to filter noise
                        speckleRange=2,  # Reduced for stricter filtering
                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                    )
                    print("Loaded calibration files successfully")
                    
                    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
                    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
                    wls_filter.setLambda(8000)
                    wls_filter.setSigmaColor(1.5)
                except Exception as e:
                    print(f"Error loading calibration: {e}")
                    ready = False
            else:
                error('Not calibrated', False)
    # Depth map
    elif key == ord('d'):
        model_type = 'DPT_Hybrid'
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

# Cleanup
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()