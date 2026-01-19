#!/usr/bin/env -S python
#region - Setup
#region -- Initialization
#region --- Needed packages
import cv2                                                                  # OpenCV - Image generation
import numpy as np                                                          # Numpy - Data processing
import threading                                                            # Threading - For multi-threading
import queue                                                                # Queue - For safe multi-threading
from typing import cast, List, Optional, Dict, Any, Tuple                   # Typing
from numpy.typing import NDArray                                            # - Numpy Array type
import json                                                                 # JSON - For calibration data
import os                                                                   # OS - For file system
#endregion
#region --- Argument parsing
import sys                                                                  # System - For CLI arguments
from typing import TypeVar, Callable
T = TypeVar('T')
def arg(deff:T, pref:str, run:Callable[[str],T]=lambda x:x) -> T:
    return [*[run(arg[len(pref):]) for arg in sys.argv if arg.startswith(pref)],deff][0]
#endregion
#endregion
#region -- Calibration
chessboard:tuple[int,int]   = arg([9,6],'-bd=', lambda t:[int(v) for v in t.split('x')])
square_size:float           = arg(20.0, '-sq=', float)
camera_id:tuple[int,int]    = arg([0,2],'-ci=', lambda t:[int(v) for v in t.split(',')])
img_scale:float             = arg(0.5,  '-sc=', float)
calibration_out:str         = arg('',   '-co=')
baseline_cm:float           = arg(14.5, '-bl=', float)
use_midas:bool              = arg('f',  '-md') == ''
calib_imgs:tuple[List[NDArray],List[NDArray]] = [[],[]]
calib_data:Optional[Dict[str,Any]] = None
calib_results:List[Dict[str,Any]] = []
selected_test:int           = -1
#endregion
#region -- Hand Gesture Recognition
hand_gesture:bool           = arg('f',  '-hg') == ('' if len(calibration_out) else 'f')
gesture_smoothing:int       = arg(3,    '-gs=', int)
gesture_history:List[str]   = []
if hand_gesture:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands                    # MediaPipe Hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing          # MediaPipe Rendering Hand Points
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles  # MediaPipe Rendering Hand Points
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
#endregion
#region -- MiDaS Depth Estimation
if use_midas:
    import torch
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
#endregion
#region -- Audio Generation
import sounddevice as sd                                                    # SoundDevice - For audio output
import time                                                                 # Time - For period/time
sample_rate:int             = arg(44100,'-sr=', int)
block_size:int              = arg(512,  '-bs=', int)
num_bins:int                = arg(64,   '-nb=', int)
history_duration:float      = arg(10.0, '-hd=', float)
echo_duration:float         = arg(0.0,  '-ed=', float)
max_depth_meters:float      = arg(5.0,  '-dm=', float)
min_volume:float            = arg(0.01, '-mv=', float)
time_counter:tuple[float]   = [0]
time_start:float            = time.time()
frequencies = np.logspace(np.log10(100), np.log10(8000), num_bins)
time_block = np.arange(block_size) / sample_rate
audio_enabled:bool          = arg('f',  '-ae') == ''
current_gesture:Dict[str,Any] = {'type': 'none', 'radius': 0.0}
spectrogram_history = {
    'left': np.zeros((num_bins, int(history_duration * sample_rate / block_size))),
    'right': np.zeros((num_bins, int(history_duration * sample_rate / block_size))),
    'time_axis': np.linspace(-history_duration, 0, int(history_duration * sample_rate / block_size))
}
spectrogram_lock = threading.Lock()
#endregion
#region -- Graphical window
import matplotlib.pyplot as plt                                             # Matplotlib - Rendering graphs
gui_window:bool             = arg('f',  '-gw') == '' or len(calibration_out)
#endregion
#region -- Stereo Matching Parameters
stereo_num_disparities:int  = arg(160,  '-sd=', int)
stereo_block_size:int       = arg(15,   '-sb=', int)
stereo_min_disparity:int    = arg(0,    '-sm=', int)
stereo_uniqueness:int       = arg(5,    '-su=', int)
stereo_speckle_window:int   = arg(100,  '-sw=', int)
stereo_speckle_range:int    = arg(2,    '-sr2=',int)
#endregion
#endregion
#region - Functions
#region -- Logging
EXIT = 1
SUCCESS = 2
WARNING = 4
ERROR = 6
def log(code:int, msg:str) -> None:
    print(['\x1b[1;2mLog','\x1b[1;32mSuccess','\x1b[1;33mWarning','\x1b[1;31mError'][code>>1]+':\x1b[0m '+msg)
    if code&1:
        print('\x1b[2m----- EXITED -----\x1b[0m')
        exit()
#endregion
#region -- Calibration File Management
def load_calibration() -> bool:
    global calib_data, calib_results, selected_test
    if not calibration_out or not os.path.exists(f"{calibration_out}.json"):
        return False
    
    try:
        with open(f"{calibration_out}.json", 'r') as f:
            data = json.load(f)
        
        calib_results = data.get('tests', [])
        if not calib_results:
            return False
        
        selected_test = len(calib_results) - 1
        test_data = calib_results[selected_test]
        
        # Convert lists back to numpy arrays with correct dtypes
        for key in ['camera_matrix_left', 'dist_coeffs_left', 'camera_matrix_right', 
                    'dist_coeffs_right', 'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q']:
            if key in test_data:
                test_data[key] = np.array(test_data[key])
        
        # Rectification maps need specific dtypes
        if 'map1_left' in test_data:
            test_data['map1_left'] = np.array(test_data['map1_left'], dtype=np.int16)
        if 'map2_left' in test_data:
            test_data['map2_left'] = np.array(test_data['map2_left'], dtype=np.uint16)
        if 'map1_right' in test_data:
            test_data['map1_right'] = np.array(test_data['map1_right'], dtype=np.int16)
        if 'map2_right' in test_data:
            test_data['map2_right'] = np.array(test_data['map2_right'], dtype=np.uint16)
        
        calib_data = test_data
        log(SUCCESS, f"Loaded {len(calib_results)} calibration test(s)")
        return True
    except Exception as e:
        log(ERROR, f"Failed to load calibration: {str(e)}")
        return False

def save_calibration(new_calib_data: Dict[str,Any]) -> None:
    global calib_data, calib_results, selected_test
    try:
        # Convert numpy arrays to lists for JSON serialization
        save_data = {}
        for key, value in new_calib_data.items():
            if isinstance(value, np.ndarray):
                save_data[key] = value.tolist()
            else:
                save_data[key] = value
        
        calib_results.append(save_data)
        selected_test = len(calib_results) - 1
        
        # Save to file
        with open(f"{calibration_out}.json", 'w') as f:
            json.dump({'tests': calib_results}, f, indent=2)
        
        # Restore numpy arrays for current use
        for key in ['camera_matrix_left', 'dist_coeffs_left', 'camera_matrix_right', 
                    'dist_coeffs_right', 'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q',
                    'map1_left', 'map2_left', 'map1_right', 'map2_right']:
            if key in new_calib_data:
                new_calib_data[key] = np.array(new_calib_data[key])
        
        calib_data = new_calib_data
        log(SUCCESS, f"Saved calibration test {selected_test+1}")
    except Exception as e:
        log(ERROR, f"Failed to save calibration: {str(e)}")
#endregion
#region -- Stereo Calibration
def run_stereo_calibration(frames_left: List[NDArray], frames_right: List[NDArray], 
                          img_shape: tuple[int, int]) -> Optional[Dict[str,Any]]:
    if len(frames_left) < 5:
        log(WARNING, "Need at least 5 calibration images")
        return None
    
    log(0, "Processing calibration images...")
    
    # Prepare object points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp *= (square_size / 1000.0)  # Convert to meters
    
    obj_points: List[NDArray] = []
    img_points_left: List[NDArray] = []
    img_points_right: List[NDArray] = []
    
    # Find chessboard corners in all frames
    for i, (frame_l, frame_r) in enumerate(zip(frames_left, frames_right)):
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard, None)
        
        if ret_l and ret_r:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            
            obj_points.append(objp)
            img_points_left.append(corners_l)
            img_points_right.append(corners_r)
    
    if len(obj_points) < 5:
        log(WARNING, "Not enough valid calibration images")
        return None
    
    log(0, f"Calibrating with {len(obj_points)} valid image pairs...")
    
    # Calibrate individual cameras
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        obj_points, img_points_left, img_shape, None, None)
    
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        obj_points, img_points_right, img_shape, None, None)
    
    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        mtx_l, dist_l, mtx_r, dist_r,
        img_shape, criteria=criteria, flags=flags)
    
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
    
    return {
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

def compute_depth_map_stereo(frame_left: NDArray, frame_right: NDArray) -> Tuple[NDArray, NDArray]:
    if calib_data is None:
        return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)
    
    # Rectify images
    rect_left = cv2.remap(frame_left, calib_data['map1_left'], 
                         calib_data['map2_left'], cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, calib_data['map1_right'], 
                          calib_data['map2_right'], cv2.INTER_LINEAR)
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    
    # Create stereo matcher with configurable parameters
    stereo = cv2.StereoBM_create(
        numDisparities=stereo_num_disparities,
        blockSize=stereo_block_size
    )
    stereo.setMinDisparity(stereo_min_disparity)
    stereo.setUniquenessRatio(stereo_uniqueness)
    stereo.setSpeckleWindowSize(stereo_speckle_window)
    stereo.setSpeckleRange(stereo_speckle_range)
    
    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # Convert disparity to depth (in meters)
    focal_length = calib_data['camera_matrix_left'][0, 0]
    baseline = calib_data['baseline_cm'] / 100.0  # Convert to meters
    
    # Avoid division by zero
    depth_map = np.zeros_like(disparity)
    valid_mask = disparity > 0
    depth_map[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
    
    # Clip depth values
    depth_map = np.clip(depth_map, 0, max_depth_meters)
    
    # Normalize and colorize for visualization
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(disparity_normalized.astype(np.uint8), 
                                     cv2.COLORMAP_JET)
    
    return depth_colored, depth_map

def compute_depth_map_midas(frame: NDArray) -> Tuple[NDArray, NDArray]:
    # Prepare input
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)
    
    # Predict depth
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map (MiDaS outputs inverse depth)
    depth_map = cv2.normalize(depth_map, None, 0, max_depth_meters, cv2.NORM_MINMAX)
    depth_map = max_depth_meters - depth_map  # Invert so closer is smaller
    
    # Create colored visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), 
                                     cv2.COLORMAP_JET)
    
    return depth_colored, depth_map
#endregion
#region -- Hand Gesture Recognition
def detect_hand_gesture(frame: NDArray) -> Dict[str, Any]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    gesture = {'type': 'none', 'radius': 0.0, 'landmarks': None}
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check if it's the right hand
            if handedness.classification[0].label == 'Right':
                gesture['landmarks'] = hand_landmarks
                
                # Get key landmarks
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                
                # Calculate if fingers are extended
                index_extended = index_tip.y < index_mcp.y
                middle_extended = middle_tip.y < middle_mcp.y
                ring_extended = ring_tip.y < ring_mcp.y
                pinky_extended = pinky_tip.y < pinky_mcp.y
                
                # Calculate openness (0 = closed fist, 1 = open hand)
                finger_extensions = [index_extended, middle_extended, ring_extended, pinky_extended]
                openness = sum(finger_extensions) / 4.0
                
                # Check if backhand is visible (z-coordinate indicates depth)
                # Positive z means further from camera (backhand visible)
                avg_z = np.mean([lm.z for lm in hand_landmarks.landmark])
                is_backhand = avg_z > -0.02  # Threshold for backhand detection
                
                if is_backhand:
                    gesture['type'] = 'backhand'
                    gesture['radius'] = openness
                else:
                    gesture['type'] = 'palm'
    
    return gesture

def smooth_gesture(gesture: Dict[str, Any]) -> Dict[str, Any]:
    global gesture_history
    
    gesture_history.append(gesture['type'])
    if len(gesture_history) > gesture_smoothing:
        gesture_history.pop(0)
    
    # Use majority voting for gesture type
    if len(gesture_history) >= gesture_smoothing:
        most_common = max(set(gesture_history), key=gesture_history.count)
        gesture['type'] = most_common
    
    return gesture
#endregion
#region -- Audio Generation from Depth Map
def depth_to_audio_params(depth_map: NDArray, radius: float) -> Tuple[NDArray, NDArray]:
    h, w = depth_map.shape
    center_x, center_y = w // 2, h // 2
    
    # Calculate the region of interest based on radius
    if radius == 0:
        # Only center point
        roi_size = 1
    else:
        # Scale from center to full width
        roi_size = int(w * radius / 2)
    
    # Extract circular region
    y_indices, x_indices = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    if radius == 0:
        mask = (x_indices == center_x) & (y_indices == center_y)
    else:
        mask = dist_from_center <= roi_size
    
    # Get depth values in the region
    masked_depth = np.where(mask, depth_map, 0)
    
    # Create spatial bins (left to right)
    spatial_bins = np.zeros((num_bins, 2))  # [left_channel, right_channel]
    
    for i in range(num_bins):
        # Calculate horizontal slice
        bin_left = int(center_x - roi_size + (2 * roi_size * i / num_bins))
        bin_right = int(center_x - roi_size + (2 * roi_size * (i + 1) / num_bins))
        bin_left = max(0, min(w - 1, bin_left))
        bin_right = max(0, min(w - 1, bin_right))
        
        # Calculate vertical position (for frequency)
        bin_top = int(center_y - roi_size * 0.5)
        bin_bottom = int(center_y + roi_size * 0.5)
        bin_top = max(0, min(h - 1, bin_top))
        bin_bottom = max(0, min(h - 1, bin_bottom))
        
        # Extract depth values in this spatial bin
        if bin_right > bin_left and bin_bottom > bin_top:
            bin_depths = masked_depth[bin_top:bin_bottom, bin_left:bin_right]
            
            if bin_depths.size > 0 and np.any(bin_depths > 0):
                # Get median depth (filter out zeros)
                valid_depths = bin_depths[bin_depths > 0]
                if valid_depths.size > 0:
                    median_depth = np.median(valid_depths)
                    
                    # Convert depth to volume (closer = louder)
                    # Normalize to 0-1 range
                    volume = 1.0 - (median_depth / max_depth_meters)
                    volume = max(min_volume, min(1.0, volume))
                    
                    # Calculate stereo position (-1 = left, 0 = center, 1 = right)
                    bin_center = (bin_left + bin_right) / 2
                    stereo_pos = (bin_center - center_x) / (w / 2)
                    
                    # Convert stereo position to left/right channels
                    if stereo_pos < 0:
                        # Left side
                        spatial_bins[i, 0] = volume * (1.0 + stereo_pos)  # Left channel louder
                        spatial_bins[i, 1] = volume * abs(stereo_pos)      # Right channel quieter
                    else:
                        # Right side
                        spatial_bins[i, 0] = volume * (1.0 - stereo_pos)  # Left channel quieter
                        spatial_bins[i, 1] = volume * stereo_pos          # Right channel louder
    
    # Apply equal loudness contour compensation
    # Higher frequencies need slight boost to be perceived as equal volume
    loudness_compensation = 1.0 + 0.3 * (np.arange(num_bins) / num_bins)
    spatial_bins *= loudness_compensation[:, np.newaxis]
    
    return spatial_bins[:, 0], spatial_bins[:, 1]  # left_channel, right_channel
#endregion
#region -- Capturing camera
frame_queues = [queue.Queue(maxsize=1) for _ in range(2)]
depth_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def capture_thread(cam_idx, cam):
    while not stop_event.is_set():
        cam.grab()
        ret, frame = cam.retrieve()
        if ret:
            if img_scale != 1.0:
                frame = cv2.resize(frame, (width, height))
            try:
                frame_queues[cam_idx].put_nowait(frame)
            except queue.Full:
                pass

def depth_processing_thread():
    global current_gesture
    
    while not stop_event.is_set():
        # Get frames from both cameras
        frames = [None, None]
        for i in range(2):
            try:
                frames[i] = frame_queues[i].get(timeout=0.1)
            except queue.Empty:
                continue
        
        if frames[0] is None or frames[1] is None:
            continue
        
        # Compute depth map
        if use_midas:
            depth_colored, depth_map = compute_depth_map_midas(frames[0])
        else:
            depth_colored, depth_map = compute_depth_map_stereo(frames[0], frames[1])
        
        # Detect hand gesture
        if hand_gesture:
            gesture = detect_hand_gesture(frames[1])
            gesture = smooth_gesture(gesture)
            
            with audio_lock:
                current_gesture = gesture
        
        # Put depth map in queue for visualization
        try:
            depth_queue.put_nowait((depth_colored, depth_map))
        except queue.Full:
            pass
#endregion
#region -- Audio Generation Callback
phases = [np.zeros(num_bins) for _ in range(2)]
current_amplitudes = [np.zeros(num_bins) for _ in range(2)]
target_amplitudes = [np.zeros(num_bins) for _ in range(2)]
smoothing_factor = 0.1

# def audio_callback(outdata, frames, time_info, status):
#     # global current_amplitudes, target_amplitudes
    
#     # if status:
#     #     print(f"Audio Status: {status}")
    
#     # time_counter[0] += block_size / sample_rate
#     # audio = [np.zeros(block_size) for _ in range(2)]
    
#     # # Get current gesture and depth map
#     # with audio_lock:
#     #     gesture = current_gesture.copy()
    
#     # # Update target amplitudes based on gesture
#     # if gesture['type'] == 'backhand':
#     #     # Get latest depth map
#     #     try:
#     #         _, depth_map = depth_queue.get_nowait()
#     #         left_amps, right_amps = depth_to_audio_params(depth_map, gesture['radius'])
#     #         target_amplitudes[0] = left_amps
#     #         target_amplitudes[1] = right_amps
#     #     except queue.Empty:
#     #         pass
#     # else:
#     #     # No sound - fade to zero
#     #     target_amplitudes = [np.zeros(num_bins) for _ in range(2)]
    
#     # # Smooth amplitude changes to avoid clicks
#     # for n in range(2):
#     #     current_amplitudes[n] += (target_amplitudes[n] - current_amplitudes[n]) * smoothing_factor
    
#     # # Generate audio from frequency bins
#     # t_buff = np.arange(block_size) / sample_rate
    
#     # for i, freq in enumerate(frequencies):
#     #     phase_increment = 2 * np.pi * freq * t_buff
        
#     #     for n in range(2):
#     #         if current_amplitudes[n][i] > min_volume:
#     #             audio[n] += current_amplitudes[n][i] * np.sin(phase_increment + phases[n][i])
#     #             phases[n][i] = (phases[n][i] + 2 * np.pi * freq * block_size / sample_rate) % (2 * np.pi)
    
#     # # Apply echo effect if enabled
#     # if echo_duration != 0 and gesture['type'] == 'backhand':
#     #     echo_samples = int(abs(echo_duration) * sample_rate)
#     #     if echo_samples > 0 and ech
#     if status: print(f"Status: {status}")
    
#     global current_amplitudes, target_amplitudes, phases
    
#     # Get current gesture and depth map
#     with audio_lock:
#         gesture = current_gesture.copy()
    
#     try:
#         depth_colored, depth_map = depth_queue.get_nowait()
        
#         # Only generate audio for backhand gesture
#         if gesture['type'] == 'backhand':
#             left_amps, right_amps = depth_to_audio_params(depth_map, gesture['radius'])
#             target_amplitudes[0] = left_amps
#             target_amplitudes[1] = right_amps
#         else:
#             target_amplitudes[0] = np.zeros(num_bins)
#             target_amplitudes[1] = np.zeros(num_bins)
#     except queue.Empty:
#         target_amplitudes[0] = np.zeros(num_bins)
#         target_amplitudes[1] = np.zeros(num_bins)
    
#     # Smooth amplitude transitions
#     for ch in range(2):
#         current_amplitudes[ch] += (target_amplitudes[ch] - current_amplitudes[ch]) * smoothing_factor
    
#     # Generate audio
#     audio = [np.zeros(block_size) for _ in range(2)]
#     t_buff = np.arange(block_size) / sample_rate
    
#     # Apply echo effect if enabled
#     if echo_duration != 0:
#         echo_samples = int(abs(echo_duration) * sample_rate)
#         if echo_samples > block_size:
#             echo_samples = block_size
        
#         for i, freq in enumerate(frequencies):
#             phase_increment = 2 * np.pi * freq * t_buff
            
#             for ch in range(2):
#                 wave = current_amplitudes[ch][i] * np.sin(phase_increment + phases[ch][i])
                
#                 if echo_duration > 0:
#                     # Sweep from left to right
#                     start_ch = 0 if ch == 0 else 1
#                     end_ch = 1 if ch == 0 else 0
                    
#                     for s in range(block_size):
#                         progress = s / echo_samples if echo_samples > 0 else 1
#                         progress = min(1.0, progress)
                        
#                         if ch == start_ch:
#                             audio[0][s] += wave[s] * (1 - progress)
#                             audio[1][s] += wave[s] * progress
#                         else:
#                             audio[1][s] += wave[s] * (1 - progress)
#                             audio[0][s] += wave[s] * progress
#                 elif echo_duration < 0:
#                     # Sweep from right to left
#                     start_ch = 1 if ch == 0 else 0
#                     end_ch = 0 if ch == 0 else 1
                    
#                     for s in range(block_size):
#                         progress = s / echo_samples if echo_samples > 0 else 1
#                         progress = min(1.0, progress)
                        
#                         if ch == start_ch:
#                             audio[1][s] += wave[s] * (1 - progress)
#                             audio[0][s] += wave[s] * progress
#                         else:
#                             audio[0][s] += wave[s] * (1 - progress)
#                             audio[1][s] += wave[s] * progress
#                 else:
#                     audio[ch] += wave
                
#                 phases[ch][i] = (phases[ch][i] + 2*np.pi*freq*block_size/sample_rate) % (2*np.pi)
#     else:
#         # Normal stereo output without echo
#         for i, freq in enumerate(frequencies):
#             phase_increment = 2 * np.pi * freq * t_buff
            
#             for ch in range(2):
#                 audio[ch] += current_amplitudes[ch][i] * np.sin(phase_increment + phases[ch][i])
#                 phases[ch][i] = (phases[ch][i] + 2*np.pi*freq*block_size/sample_rate) % (2*np.pi)
    
#     # Normalize to prevent clipping
#     max_val = max(np.abs(aud).max() for aud in audio)
#     if max_val > 0:
#         audio = [aud / max_val * 0.8 for aud in audio]  # 0.8 for safety margin
    
#     outdata[:] = np.column_stack(audio).astype(np.float32)[:frames]
def audio_callback(outdata, frames, status):
    if status: print(f"Status: {status}")
    
    global current_amplitudes, target_amplitudes, phases, spectrogram_history
    
    # Get current gesture and depth map
    with audio_lock:
        gesture = current_gesture.copy()
    
    try:
        depth_colored, depth_map = depth_queue.get_nowait()
        
        # Only generate audio for backhand gesture
        if gesture['type'] == 'backhand':
            left_amps, right_amps = depth_to_audio_params(depth_map, gesture['radius'])
            target_amplitudes[0] = left_amps
            target_amplitudes[1] = right_amps
        else:
            target_amplitudes[0] = np.zeros(num_bins)
            target_amplitudes[1] = np.zeros(num_bins)
    except queue.Empty:
        target_amplitudes[0] = np.zeros(num_bins)
        target_amplitudes[1] = np.zeros(num_bins)
    
    # Smooth amplitude transitions
    for ch in range(2):
        current_amplitudes[ch] += (target_amplitudes[ch] - current_amplitudes[ch]) * smoothing_factor
    
    # Update spectrogram history
    with spectrogram_lock:
        spectrogram_history['left'] = np.roll(spectrogram_history['left'], -1, axis=1)
        spectrogram_history['right'] = np.roll(spectrogram_history['right'], -1, axis=1)
        spectrogram_history['left'][:, -1] = current_amplitudes[0]
        spectrogram_history['right'][:, -1] = current_amplitudes[1]
    
    # Generate audio
    audio = [np.zeros(block_size) for _ in range(2)]
    t_buff = np.arange(block_size) / sample_rate
    
    # Apply echo effect if enabled
    if echo_duration != 0:
        echo_samples = int(abs(echo_duration) * sample_rate)
        if echo_samples > block_size:
            echo_samples = block_size
        
        for i, freq in enumerate(frequencies):
            phase_increment = 2 * np.pi * freq * t_buff
            
            for ch in range(2):
                wave = current_amplitudes[ch][i] * np.sin(phase_increment + phases[ch][i])
                
                if echo_duration > 0:
                    # Sweep from left to right
                    start_ch = 0 if ch == 0 else 1
                    end_ch = 1 if ch == 0 else 0
                    
                    for s in range(block_size):
                        progress = s / echo_samples if echo_samples > 0 else 1
                        progress = min(1.0, progress)
                        
                        if ch == start_ch:
                            audio[0][s] += wave[s] * (1 - progress)
                            audio[1][s] += wave[s] * progress
                        else:
                            audio[1][s] += wave[s] * (1 - progress)
                            audio[0][s] += wave[s] * progress
                elif echo_duration < 0:
                    # Sweep from right to left
                    start_ch = 1 if ch == 0 else 0
                    end_ch = 0 if ch == 0 else 1
                    
                    for s in range(block_size):
                        progress = s / echo_samples if echo_samples > 0 else 1
                        progress = min(1.0, progress)
                        
                        if ch == start_ch:
                            audio[1][s] += wave[s] * (1 - progress)
                            audio[0][s] += wave[s] * progress
                        else:
                            audio[0][s] += wave[s] * (1 - progress)
                            audio[1][s] += wave[s] * progress
                else:
                    audio[ch] += wave
                
                phases[ch][i] = (phases[ch][i] + 2*np.pi*freq*block_size/sample_rate) % (2*np.pi)
    else:
        # Normal stereo output without echo
        for i, freq in enumerate(frequencies):
            phase_increment = 2 * np.pi * freq * t_buff
            
            for ch in range(2):
                audio[ch] += current_amplitudes[ch][i] * np.sin(phase_increment + phases[ch][i])
                phases[ch][i] = (phases[ch][i] + 2*np.pi*freq*block_size/sample_rate) % (2*np.pi)
    
    # Normalize to prevent clipping
    max_val = max(np.abs(aud).max() for aud in audio)
    if max_val > 0:
        audio = [aud / max_val * 0.8 for aud in audio]  # 0.8 for safety margin
    
    outdata[:] = np.column_stack(audio).astype(np.float32)[:frames]
#endregion
#endregion
#region - Initializing
#region -- Help menu
if arg('f', '-h') == '':
    print("""
Thesis - Sensory Substitution Device for Visual Impairments
Usage: ./thesis.py [flags]

Primary Flags:
  -h              Show this help menu
  -gw             Enable graphical window
  -co=PATH        Enable calibration mode and set output path
  
Calibration Settings:
  -bd=WxH         Chessboard dimensions (default: 9x6)
  -sq=SIZE        Square size in mm (default: 20.0)
  -bl=CM          Baseline distance in cm (default: 14.5)
  
Camera Settings:
  -ci=L,R         Camera IDs for left,right (default: 0,2)
  -sc=SCALE       Image scale factor (default: 0.5)
  
Depth Generation:
  -md             Use MiDaS instead of stereo vision
  -sd=NUM         Stereo num disparities (default: 160)
  -sb=SIZE        Stereo block size (default: 15)
  -sm=MIN         Stereo min disparity (default: 0)
  -su=RATIO       Stereo uniqueness ratio (default: 5)
  -sw=SIZE        Stereo speckle window size (default: 100)
  -sr2=RANGE      Stereo speckle range (default: 2)
  
Hand Gesture:
  -hg             Enable hand gesture recognition
  -gs=FRAMES      Gesture smoothing frames (default: 3)
  
Audio Settings:
  -ae             Enable audio output
  -sr=RATE        Sample rate in Hz (default: 44100)
  -bs=SIZE        Block size (default: 512)
  -nb=BINS        Number of frequency bins (default: 64)
  -ed=SECS        Echo duration in seconds (default: 0.0)
                  Positive = left to right, Negative = right to left
  -dm=METERS      Max depth in meters (default: 5.0)
  -mv=VOL         Minimum volume (default: 0.01)
  -hd=SECS        History duration for visualization (default: 10.0)

Examples:
  ./thesis.py -gw -ae -hg              # Run with GUI, audio, and gestures
  ./thesis.py -co=calib_data -gw       # Calibration mode
  ./thesis.py -md -gw -ae              # Use MiDaS for depth estimation
  ./thesis.py -ed=0.5 -ae              # Enable echo effect left to right
""")
    exit()
#endregion
#region -- Connecting to cameras
log(0, 'Connecting to cameras')
cameras = [cv2.VideoCapture(n) for n in camera_id]
if err:=sum((not cam.isOpened())<<n for n, cam in enumerate(cameras)):
    log(ERROR|EXIT, f"Could not open {['','left','right','both'][err]} camera/s")
log(0, 'Eliminating camera buffer')
for camera in cameras: 
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#endregion
#region -- Checking camera dimensions
log(0, 'Getting camera dimensions')
init_frames = cast(tuple[tuple[bool,NDArray[np.uint8]],tuple[bool,NDArray[np.uint8]]], [camera.read() for camera in cameras])
if err:=sum((not res[0])<<n for n, res in enumerate(init_frames)):
    log(ERROR|EXIT, f"Could not read {['','left','right','both'][err]} camera/s")
if err:=sum((init_frames[0][1].shape[:2][n] != init_frames[1][1].shape[:2][n])<<n for n in range(2)):
    log(ERROR|EXIT, f"Camera {['','width', 'height', 'dimension']} doesn't match")
height, width = [int(dim*img_scale) for dim in init_frames[0][1].shape[:2]]
log(SUCCESS, f"Camera dimensions: {width}x{height}")
#endregion
#region -- Load existing calibration
if calibration_out:
    load_calibration()
    if not calib_data and not use_midas:
        log(WARNING, "No calibration data loaded. Stereo depth will not work until calibration is complete.")
#endregion
#region -- Start audio
if audio_enabled:
    log(0, 'Starting audio stream')
    try:
        audio_stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=2,
            callback=audio_callback,
            dtype=np.float32
        )
        audio_stream.start()
        log(SUCCESS, 'Audio stream started')
    except Exception as e:
        log(ERROR, f"Failed to start audio: {str(e)}")
        audio_enabled = False
#endregion
#region -- Start depth processing thread
if not calibration_out:
    log(0, 'Starting depth processing thread')
    depth_thread = threading.Thread(target=depth_processing_thread, daemon=True)
    depth_thread.start()
#endregion
#region -- Window
if gui_window:
    #region --- Window Initialization
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FuncAnimation
    log(0, 'Creating graphical window')
    plt.ion()
    fig = plt.figure(figsize=(16,9), num='Thesis')
# Layout: 2x2 grid with 3 rows (cameras, depth/status, spectrograms)
    gs = GridSpec(3, 2, height_ratios=[3, 1, 1.5], hspace=0.15, wspace=0.15)
    
    # First Row - Cameras
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    camera_axs = [fig.add_subplot(gs[0, n]) for n in range(2)]
    camera_imgs = [camera_axs[n].imshow(placeholder) for n in range(2)]
    for ax in camera_axs:
        ax.set_aspect('equal')
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.axis('off')
    camera_axs[0].set_title('Left Camera')
    camera_axs[1].set_title('Right Camera')
    
    # Second Row - Calibration/Depth and Audio/Info
    brow = [fig.add_subplot(gs[1, n]) for n in range(2)]
    
    if len(calibration_out):
        brow[0].set_title('Calibration Error')
        brow[0].set_xlabel('Test Number', fontsize=10)
        brow[0].set_ylabel('Reprojection Error (px)', fontsize=10)
        brow[0].grid(True, alpha=0.3)
        
        brow[1].set_title('Depth Map')
        brow[1].axis('off')
        depth_img = brow[1].imshow(placeholder)
    else:
        brow[0].set_title('Depth Map')
        brow[0].axis('off')
        depth_img = brow[0].imshow(placeholder)
        
        brow[1].set_title('System Status')
        brow[1].axis('off')
        status_text = brow[1].text(0.1, 0.5, '', fontsize=12, verticalalignment='center',
                                   family='monospace')
    
    # Third Row - Spectrograms
    spectrogram_axs = [fig.add_subplot(gs[2, n]) for n in range(2)]
    spectrogram_imgs = []
    
    for i, ax in enumerate(spectrogram_axs):
        channel_name = ['Left', 'Right'][i]
        ax.set_title(f'{channel_name} Channel Spectrogram')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        
        # Create initial spectrogram image
        spec_data = np.zeros((num_bins, int(history_duration * sample_rate / block_size)))
        img = ax.imshow(spec_data, aspect='auto', origin='lower', 
                       extent=[spectrogram_history['time_axis'][0], 
                              spectrogram_history['time_axis'][-1],
                              frequencies[0], frequencies[-1]],
                       cmap='viridis', vmin=0, vmax=1, interpolation='bilinear')
        spectrogram_imgs.append(img)
        
        ax.set_yscale('log')
        ax.set_ylim([frequencies[0], frequencies[-1]])
        ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle('Thesis Calibration' if len(calibration_out) else 'Thesis Demo', fontsize=16)
    
    # Start capture threads
    threads = [threading.Thread(target=capture_thread, args=(i, cam), daemon=True) 
               for i, cam in enumerate(cameras)]
    for t in threads: t.start()
    
    # Keyboard state
    key_pressed = [None]
    def on_key(event):
        key_pressed[0] = event.key
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    log(SUCCESS, 'Graphical window created')
    log(0, 'Starting main loop')
    #endregion
    #region --- Main loop thread
    def update_animation(frame_num):
        global calib_data
        #region ---- Capturing camera
        cam_frames = [None] * 2
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries and any(f is None for f in cam_frames):
            for i in range(2):
                if cam_frames[i] is None:
                    try:
                        cam_frames[i] = frame_queues[i].get(timeout=0.1)
                    except queue.Empty:
                        pass
            retry_count += 1
        
        err = sum((cam_frames[n] is None)<<n for n in range(2))
        if err:
            log(WARNING,f"Failed to capture image from {['','left','right','both'][err]} camera/s")
        #endregion
        #region ---- Hand gesture recognition and visualization
        if hand_gesture and cam_frames[1] is not None and not calibration_out:
            # frame_rgb = cv2.cvtColor(cam_frames[1], cv2.COLOR_BGR2RGB)
            # results = hands.process(frame_rgb)
            
            # if results.multi_hand_landmarks and results.multi_handedness:
            #     for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            #         if handedness.classification[0].label == 'Right':
            #             mp_drawing.draw_landmarks(
            #                 frame_rgb,
            #                 hand_landmarks,
            #                 cast(List[tuple[int, int]], mp_hands.HAND_CONNECTIONS),
            #                 mp_drawing_styles.get_default_hand_landmarks_style(),
            #                 mp_drawing_styles.get_default_hand_connections_style()
            #             )
            # cam_frames[1] = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            frame_rgb = cv2.cvtColor(cam_frames[1], cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label == 'Right':
                        mp_drawing.draw_landmarks(
                            frame_rgb,
                            hand_landmarks,
                            cast(List[tuple[int, int]], mp_hands.HAND_CONNECTIONS),
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
            cam_frames[1] = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        #endregion
        #region ---- Calibration training
        if len(calibration_out) and not err:
            #region ----- Handle keyboard input
            key = key_pressed[0]
            key_pressed[0] = None
            
            if key == ' ':  # SPACE - Capture calibration
                gray_l = cv2.cvtColor(cam_frames[0], cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(cam_frames[1], cv2.COLOR_BGR2GRAY)
                
                ret_l, _ = cv2.findChessboardCorners(gray_l, chessboard, None)
                ret_r, _ = cv2.findChessboardCorners(gray_r, chessboard, None)
                
                if ret_l and ret_r:
                    calib_imgs[0].append(cam_frames[0].copy())
                    calib_imgs[1].append(cam_frames[1].copy())
                    
                    # Save images
                    test_num = len(calib_results) + 1
                    img_num = len(calib_imgs[0])
                    os.makedirs(f"{calibration_out}/{test_num}", exist_ok=True)
                    cv2.imwrite(f"{calibration_out}/{test_num}/L{img_num}.png", cam_frames[0])
                    cv2.imwrite(f"{calibration_out}/{test_num}/R{img_num}.png", cam_frames[1])
                    
                    log(SUCCESS, f"Captured frame {img_num}")
                else:
                    log(WARNING, "Pattern not detected in both cameras")
            
            elif key == 'enter' and len(calib_imgs[0]) > 0:  # ENTER - Process calibration
                log(0, "Processing calibration...")
                result = run_stereo_calibration(calib_imgs[0], calib_imgs[1], (width, height))
                
                if result is not None:
                    save_calibration(result)
                    calib_imgs[0].clear()
                    calib_imgs[1].clear()
                    
                    # Update graph
                    if len(calib_results) > 0:
                        brow[0].clear()
                        brow[0].set_title('Calibration Error')
                        brow[0].set_xlabel('Test Number', fontsize=10)
                        brow[0].set_ylabel('Reprojection Error (px)', fontsize=10)
                        
                        errors = [r['reprojection_error'] for r in calib_results]
                        brow[0].plot(range(1, len(errors)+1), errors, 'bo-', linewidth=2, markersize=8)
                        brow[0].grid(True, alpha=0.3)
            
            elif key == 'r' and len(calib_results) > 1:  # R - Remove last calibration
                calib_results.pop()
                selected_test = len(calib_results) - 1
                if calib_results:
                    calib_data = calib_results[selected_test]
                else:
                    calib_data = None
                
                with open(f"{calibration_out}.json", 'w') as f:
                    json.dump({'tests': calib_results}, f, indent=2)
                
                log(WARNING, f"Removed last calibration. {len(calib_results)} remaining")
                
                # Update graph
                if len(calib_results) > 0:
                    brow[0].clear()
                    brow[0].set_title('Calibration Error')
                    brow[0].set_xlabel('Test Number', fontsize=10)
                    brow[0].set_ylabel('Reprojection Error (px)', fontsize=10)
                    
                    errors = [r['reprojection_error'] for r in calib_results]
                    brow[0].plot(range(1, len(errors)+1), errors, 'bo-', linewidth=2, markersize=8)
                    brow[0].grid(True, alpha=0.3)
            #endregion
            #region ----- Chessboard detection and display
            rets = [False, False]
            corners = [None, None]
            
            for n in range(2):
                gray = cv2.cvtColor(cam_frames[n], cv2.COLOR_BGR2GRAY)
                ret, corner = cv2.findChessboardCorners(gray, chessboard, None)
                rets[n] = ret
                corners[n] = corner
                
                if ret:
                    cv2.drawChessboardCorners(cam_frames[n], chessboard, corner, ret)
            
            status = "READY" if all(rets) else "NO PATTERN"
            color = (0, 255, 0) if all(rets) else (0, 0, 255)
            
            for n in range(2):
                cv2.putText(cam_frames[0], f"CALIBRATION: {status}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color if n else (0,0,0), 3-n)
                cv2.putText(cam_frames[0], f"Captured: {len(calib_imgs[0])}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*n,)*3, 3-n)
                cv2.putText(cam_frames[0], "SPACE=Capture, ENTER=Process, R=Remove", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*n,)*3, 3-n)
            #endregion
            #region ----- Update depth map if calibration exists
            if calib_data is not None:
                depth_colored, _ = compute_depth_map_stereo(cam_frames[0], cam_frames[1])
                depth_img.set_data(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
            #endregion
        #endregion
        #region ---- Normal operation mode
        if not calibration_out:
            # Update depth map
            try:
                depth_colored, depth_map = depth_queue.get_nowait()
                depth_img.set_data(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
            except queue.Empty:
                pass
            
            # Update status text
            with audio_lock:
                gesture = current_gesture.copy()
            
            status_str = f"Gesture: {gesture['type'].upper()}\n"
            if gesture['type'] == 'backhand':
                status_str += f"Radius: {gesture['radius']:.2f}\n"
            status_str += f"\nAudio: {'ON' if audio_enabled else 'OFF'}\n"
            status_str += f"Depth: {'MiDaS' if use_midas else 'Stereo'}\n"
            status_str += f"Calibrated: {'Yes' if calib_data else 'No'}\n"
            status_str += f"\nControls:\n"
            status_str += f"  Show backhand to generate audio\n"
            status_str += f"  Open hand = wide view (radius 1.0)\n"
            status_str += f"  Close fist = narrow view (radius 0.0)\n"
            status_str += f"  Hide hand = silence\n"
            
            status_text.set_text(status_str)
        #endregion
        #region ---- Update spectrograms
        if not calibration_out:
            with spectrogram_lock:
                left_spec = spectrogram_history['left'].copy()
                right_spec = spectrogram_history['right'].copy()
            
            # Update spectrogram images
            spectrogram_imgs[0].set_data(left_spec)
            spectrogram_imgs[1].set_data(right_spec)
            
            # Auto-scale color limits based on current data
            max_amp = max(left_spec.max(), right_spec.max(), 0.01)
            for img in spectrogram_imgs:
                img.set_clim(0, max_amp)
        #endregion
        #region ---- Output
        for i in range(2):
            if cam_frames[i] is not None:
                rgb_frame = cv2.cvtColor(cam_frames[i], cv2.COLOR_BGR2RGB)
                camera_imgs[i].set_data(rgb_frame)
        
        return_list = camera_imgs.copy()
        if len(calibration_out) and calib_data is not None:
            return_list.append(depth_img)
        elif not calibration_out:
            return_list.extend([depth_img, status_text])
            return_list.extend(spectrogram_imgs)
        return return_list
        #endregion
    #endregion
    #region --- Animation and cleanup
    ani = FuncAnimation(fig, update_animation, interval=10, blit=True, cache_frame_data=False)
    
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        log(0, "Interrupted by user")
    
    # Cleanup
    log(0, "Cleaning up...")
    stop_event.set()
    
    if audio_enabled:
        audio_stream.stop()
        audio_stream.close()
    
    for t in threads: 
        t.join(timeout=1.0)
    
    if not calibration_out and depth_thread.is_alive():
        depth_thread.join(timeout=1.0)
    
    for q in frame_queues: 
        q.queue.clear()
    
    depth_queue.queue.clear()
    
    for camera in cameras:
        camera.release()
    
    if hand_gesture:
        hands.close()
    
    cv2.destroyAllWindows()
    
    log(SUCCESS, "Cleanup complete")
    #endregion
#endregion
#region -- Non-GUI mode
else:
    log(0, "Running in headless mode")
    log(0, "Press Ctrl+C to stop")
    
    # Start capture threads
    threads = [threading.Thread(target=capture_thread, args=(i, cam), daemon=True) 
               for i, cam in enumerate(cameras)]
    for t in threads: t.start()
    
    try:
        while True:
            time.sleep(0.1)
            
            # Optional: Print status periodically
            with audio_lock:
                gesture = current_gesture.copy()
            
            if gesture['type'] == 'backhand':
                print(f"\rGesture: {gesture['type']} | Radius: {gesture['radius']:.2f}  ", end='')
            else:
                print(f"\rGesture: {gesture['type']}                    ", end='')
    
    except KeyboardInterrupt:
        log(0, "\nInterrupted by user")
    
    # Cleanup
    log(0, "Cleaning up...")
    stop_event.set()
    
    if audio_enabled:
        audio_stream.stop()
        audio_stream.close()
    
    for t in threads:
        t.join(timeout=1.0)
    
    if depth_thread.is_alive():
        depth_thread.join(timeout=1.0)
    
    for q in frame_queues:
        q.queue.clear()
    
    depth_queue.queue.clear()
    
    for camera in cameras:
        camera.release()
    
    if hand_gesture:
        hands.close()
    
    log(SUCCESS, "Cleanup complete")
#endregion
#endregion