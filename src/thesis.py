#!/home/jazo/py/bin/python
# Library imports: pip install "numpy>=2.0,<2.3" mediapipe opencv-python --upgrade
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# Typing imports
from numpy.typing import NDArray
from typing import cast, NamedTuple, Optional, List, Any

# Configuration Variables
board:tuple[int,int]    = (9,6)     # Inner corners of checkerboard
square_size:int         = 21        # Size of square in mm
cam_id:tuple[int,int]   = (0,2)     # Camera ID for /dev/video{n} by "v4l2-ctl --list-devices"
scale:float             = 0.5       # Scale ratio for faster processing

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

# Helper Functions
def error(msg:str, end:bool=True) -> None:
    print(f"\x1b[1;31mError:\x1b[0m {msg}")
    if end: exit()

# Calibration Variables
objp = np.zeros((board[0]*board[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:board[0],0:board[1]].T.reshape(-1,2)
objp *= square_size
objpoints:list[NDArray[np.float32]] = [] # 3D points
points_left:list[NDArray[np.float32]] = [] # 2D points from left camera
points_right:list[NDArray[np.float32]] = [] # 2D points from right camera
count = 0
gray_left:NDArray[np.uint8]|None = None
gray_right:NDArray[np.uint8]|None = None

# Hand guesture recognition Variables
hand_guesture:bool = False
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize the two cameras
print('Connecting to camera')
cam_left, cam_right = [cv2.VideoCapture(n) for n in cam_id]
if err:=(not cam_left.isOpened())+((not cam_right.isOpened())<<1):
    error(f"Could not open {['','left','right','both'][err]} camera/s")

# Elimintate buffer for most lastest frame capture
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

# Create a window
print('Starting graphical window')
cv2.namedWindow('Dual Camera Feed', cv2.WINDOW_NORMAL)

while True:
    # Read frames from both cameras
    ((ret_left, frame_left), (ret_right, frame_right)) = [cast(tuple[bool, NDArray[np.uint8]], cam.read()) for cam in [cam_left, cam_right]]
    if err:=(not ret_left)+((not ret_right)<<1):
        error(f"Could not capture video from {['','left','right','both'][err]} camera/s")

    # Resize to ensure equivelent dimensions
    frame_left = cv2.flip(cv2.resize(frame_left, (width, height)), -1)
    frame_right = cv2.flip(cv2.resize(frame_right, (width, height)), -1)

    # Hand guesture detection in right camera
    if hand_guesture:
        frame = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        results = cast(HandsResults, hands.process(frame))
        # Is there any detected hands?
        if results.multi_hand_landmarks and results.multi_handedness:
            # Render each hand
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmark
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    cast(List[tuple[int, int]], mp_hands.HAND_CONNECTIONS),
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
        # Update to frame
        frame_right = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display them side by side
    cv2.imshow('Dual Camera Feed', np.hstack((frame_left, frame_right)))

    # Key handler
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' or esc is pressed
    if key == ord('q') or key == 27:
        break

    # Space to capture for calibration
    elif key == ord('c'):
        gray_left, gray_right = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in [frame_left, frame_right]]
        ((ret_left,corners_left), (ret_right,corners_right)) = [cv2.findChessboardCorners(gray, board) for gray in [gray_left, gray_right]]
        # Checkerboard not found in both images
        if not ret_left and not ret_right:
            error('Checkerboard not found in both images', False)
        # Checkerboard found, keep track of it
        else:
            objpoints.append(objp)
            corners_left, corners_right = [cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            ) for gray, corners in [(gray_left, corners_left), (gray_right, corners_right)]]
            points_left.append(corners_left)
            points_right.append(corners_right)
            count += 1
            print(f"\x1b[1;33mNote:\x1b[0m Captured {count} calibration image/s")
    
    # Generate Calibration (https://docs.opencv.org/4.x/pattern.png)
    elif key == ord('g'):
        # Check if captured images first
        if gray_left == None or gray_right == None:
            error('No calibration images captured')
        assert gray_left is not None
        assert gray_right is not None
        # Calibrate each camera individually
        ((ret_left ,K_left ,dist_left ,_,_),
         (ret_right,K_right,dist_right,_,_)) = [cv2.calibrateCamera(
            objpoints, points, gray.shape[::-1], None, None
         ) for points, gray in [(points_left, gray_left), (points_right, gray_right)]]
        # Calibrate for stereo
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, points_left, points_right,
            K_left, dist_left, K_right, dist_right,
            gray_left.shape[::-1], criteria=criteria, flags=flags
        )
        # Save calibration data
        np.savez('stero_calibration.npz',
            K_left=K_left, dist_left=dist_left,
            K_right=K_right, dist_right=dist_right,
            R=R, T=T, E=E, F=F
        )
    
    # Compute Rectificaton map
    elif key == ord('r'):
        # Load calibrations
        calib = np.load('stereo_calibration.npz')
        K_left = calib['K_left']
        dist_left = calib['dist_left']
        K_right = calib['K_right']
        dist_right = calib['dist_right']
        R = calib['R']
        T = calib['T']

        # Compute transforms
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right,
            (width, height), R, T, alpha=0
        )

        # Compute maps
        ((map_left_x, map_left_y),(map_right_x,map_right_y)) = [cv2.initUndistortRectifyMap(
            K, dist, R, P, (width, height), cv2.CV_32FC1
        ) for K, dist, R, P in [(K_left, dist_left, R1, P1),(K_right, dist_right, R2, P2)]]

        # Save map
        np.savez('rectification_maps.npz',
            map_left_x =map_left_x , map_left_y =map_left_y,
            map_right_x=map_right_x, map_right_y=map_right_y,
            Q=Q
        )

    # Toggle hand detection
    elif key == ord('h'):
        hand_guesture = not hand_guesture


# Release the cameras and close windows
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
