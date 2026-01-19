#!/usr/bin/env -S python
#region - Setup
#region -- Intialization
#region --- Needed packages
import cv2                                                                  # OpenCV - Image generation
import numpy as np                                                          # Numpy - Data processing
import threading                                                            # Threading - For multi-threading
import queue                                                                # Queue - For safe multi-threading
from typing import cast, List, Optional, Dict, Any                          # Typing
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
calib_imgs:tuple[List[NDArray],List[NDArray]] = [[],[]]
calib_data:Optional[Dict[str,Any]] = None
calib_results:List[Dict[str,Any]] = []
selected_test:int           = -1
#endregion
#region -- Hand Gesture Recognition
hand_gesture:bool           = arg('f',  '-hg') == ('' if len(calibration_out) else 'f')
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
#region -- Audio Generation
import sounddevice as sd                                                    # SoundDevice - For audio output
import time                                                                 # Time - For period/time
sample_rate:int             = arg(44100,'-sr=', int)
block_size:int              = arg(512,  '-bs=', int)
num_bins:int                = arg(64,   '-nb=', int)
history_duration:float      = arg(10.0, '-hd=', int)
time_counter:tuple[float]   = [0]
time_start:float            = time.time()
frequencies = np.logspace(np.log10(100), np.log10(8000), num_bins)
time_block = np.arange(block_size) / sample_rate
audio_enabled:bool          = arg('f',  '-ae') == ''
#endregion
#region -- Graphical window
import matplotlib.pyplot as plt                                             # Matplotlib - Rendering graphs
gui_window:bool             = arg('f',  '-gw') == '' or len(calibration_out)
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

def compute_depth_map(frame_left: NDArray, frame_right: NDArray) -> NDArray:
    if calib_data is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
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
    
    # Normalize and colorize
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(disparity_normalized.astype(np.uint8), 
                                     cv2.COLORMAP_JET)
    return depth_colored
#endregion
#region -- Capturing camera
frame_queues = [queue.Queue(maxsize=1) for _ in range(2)]
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
#endregion
#region -- Audio Generation
phases = [np.zeros(num_bins) for _ in range(2)]
t_buff = np.arange(block_size) / sample_rate
def audio_callback(outdata, frames, time_info, status):
    if status: print(f"Status: {status}")
    time_counter[0] += 0.01
    audio = [np.zeros(block_size) for _ in range(2)]
    
    # Peak computation (temporary)
    peak_pos = (np.sin(time_counter[0]*2)+1)/2
    vars = [0.1*np.exp(-((np.arange(num_bins) - (n+(1-2*n)*peak_pos) * num_bins) ** 2) / 20) for n in range(2)]
    
    for i, freq in enumerate(frequencies):
        phase_increment = 2 * np.pi * freq * t_buff
        for n in range(2):
            audio[n] += vars[n][i] * np.sin(phase_increment + phases[n][i])
            phases[n][i] = (phases[n][i] + 2*np.pi*freq*block_size/sample_rate) % (2*np.pi)
    
    # Normalize
    max_val = max(np.abs(aud).max() for aud in audio)
    if max_val > 0: audio = [aud/max_val for aud in audio]
    
    outdata[:] = np.column_stack(audio).astype(np.float32)[:frames]
#endregion
#endregion

#region - Initalizating
#region -- Connecting to cameras
log(0, 'Connecting to cameras')
cameras = [cv2.VideoCapture(n) for n in camera_id]
if err:=sum((not cam.isOpened())<<n for n, cam in enumerate(cameras)):
    log(ERROR|EXIT, f"Could not open {['','left','right','both'][err]} camera/s")
log(0, 'Eliminating camera buffer')
for camera in cameras: camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#endregion
#region -- Checking camera dimensions
log(0, 'Getting camera dimensions')
init_frames = cast(tuple[tuple[bool,NDArray[np.uint8]],tuple[bool,NDArray[np.uint8]]], [camera.read() for camera in cameras])
if err:=sum((not res[0])<<n for n, res in enumerate(init_frames)):
    log(ERROR|EXIT, f"Could not read {['','left','right','both'][err]} camera/s")
if err:=sum((init_frames[0][1].shape[:2][n] != init_frames[1][1].shape[:2][n])<<n for n in range(2)):
    log(ERROR|EXIT, f"Camera {['','width', 'height', 'dimension']} does't match")
height, width = [int(dim*img_scale) for dim in init_frames[0][1].shape[:2]]
#endregion
#region -- Load existing calibration
if calibration_out:
    load_calibration()
#endregion
#region -- Start audio
if audio_enabled:
    audio_stream = sd.OutputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        channels=2,
        callback=audio_callback,
        dtype=np.float32
    )
    audio_stream.start()
#endregion
#region -- Window
if gui_window:
    #region --- Window Intialization
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FuncAnimation
    log(0, 'Creating graphical window')

    plt.ion()
    fig = plt.figure(figsize=(16,9), num='Thesis')
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.15, wspace=0.15)
    
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
    
    # Second Row - Calibration/Depth
    brow = [fig.add_subplot(gs[1, n]) for n in range(2)]
    if len(calibration_out):
        brow[0].set_title(f'Calibration Error')
        brow[0].set_xlabel('Test Number', fontsize=10)
        brow[0].set_ylabel('Reprojection Error (px)', fontsize=10)
        
        brow[1].set_title('Depth Map')
        brow[1].axis('off')
        depth_img = brow[1].imshow(placeholder)
    else:
        brow[0].set_title('Left Audio Channel')
        brow[0].set_xlabel('Time (seconds)', fontsize=10)
        brow[0].set_ylabel('Frequency (Hz)', fontsize=10)
        brow[1].set_title('Right Audio Channel')
        brow[1].set_xlabel('Time (seconds)', fontsize=10)
        brow[1].set_ylabel('Frequency (Hz)', fontsize=10)
    
    fig.suptitle('Thesis Calibration' if len(calibration_out) else 'Thesis Demo', fontsize=16)
    
    # Start capture threads
    threads = [threading.Thread(target=capture_thread, args=(i, cam), daemon=True) for i, cam in enumerate(cameras)]
    for t in threads: t.start()
    
    # Keyboard state
    key_pressed = [None]
    def on_key(event):
        key_pressed[0] = event.key
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    log(0, 'Starting main loop')
    #endregion
    #region --- Main loop thread
    def update_animation(frame_num):
        #region ---- Capturing camera
        cam_frames = [None] * 2
        for i in range(2):
            try:
                cam_frames[i] = frame_queues[i].get_nowait()
            except queue.Empty:
                pass
        
        err = sum((cam_frames[n] is None)<<n for n in range(2))
        if err:
            log(WARNING,f"Failed to capture image from {['','left','right','both'][err]} camera/s")
        #endregion
        #region ---- Hand gesture recognition
        if hand_gesture and not err:
            frame_rgb = cv2.cvtColor(cam_frames[1], cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
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
                        brow[0].set_title(f'Calibration Error')
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
                cv2.putText(cam_frames[0], f"CALIBRATION: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color if n else (0,0,0), 3-n)
                cv2.putText(cam_frames[0], f"Captured: {len(calib_imgs[0])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*n,)*3, 3-n)
                cv2.putText(cam_frames[0], "SPACE=Capture, ENTER=Process", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*n,)*3, 3-n)
            #endregion
            #region ----- Update depth map if calibration exists
            if calib_data is not None:
                depth_map = compute_depth_map(cam_frames[0], cam_frames[1])
                depth_img.set_data(cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB))
            #endregion
        #endregion
        #region ---- Output
        for i in range(2):
            if cam_frames[i] is not None:
                rgb_frame = cv2.cvtColor(cam_frames[i], cv2.COLOR_BGR2RGB)
                camera_imgs[i].set_data(rgb_frame)
        
        return_list = camera_imgs.copy()
        if len(calibration_out) and calib_data is not None:
            return_list.append(depth_img)
        return return_list
        #endregion
    #endregion
    #region --- Ending
    ani = FuncAnimation(fig, update_animation, interval=10, blit=True, cache_frame_data=False)
    plt.show(block=True)
    
    # Cleanup
    stop_event.set()
    for t in threads: t.join()
    for q in frame_queues: q.queue.clear()
    #endregion
#endregion
#endregion
