#region - Setup
#region -- Intialization
#region --- Needed packages
import cv2                                                                  # OpenCV - Image generation
import numpy as np                                                          # Numpy - Data processing
import threading                                                            # Threading - For multi-threading
import queue                                                                # Queue - For safe multi-threading
from typing import cast, List                                               # Typing
from numpy.typing import NDArray                                            # - Numpy Array type
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
square_size:int             = arg(19,   '-sq=', int)
camera_id:tuple[int,int]    = arg([0,2],'-ci=', lambda t:[int(v) for v in t.split(',')])
img_scale:float             = arg(0.5,  '-sc=', float)
calibration_out:str         = arg('',   '-co=')
baseline_cm:float           = arg(14.5, '-bl=', float)
calib_imgs:tuple[List[NDArray],List[NDArray]] = [[],[]]
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
# Frequency bns in logarithmic spacing for more perceptual relevance
frequencies = np.logspace(np.log10(100), np.log10(8000), num_bins)
# Time array for one block
time_block = np.arange(block_size) / sample_rate
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
#region -- Capturing camera
# Thread to capture frames and prevents blocking the animation
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
    # Time unit
    time_counter[0] += 0.01
    # Audio channel
    audio:list[NDArray[np.float64]] = [np.zeros(block_size) for _ in range(2)]
    # Peak computation (temporary)
    peak_pos = (np.sin(time_counter[0]*2)+1)/2
    vars = [0.1*np.exp(-((np.arange(num_bins) - (n+(1-2*n)*peak_pos) * num_bins) ** 2) / 20) for n in range(2)]
    # Per frequency processing
    for i, freq in enumerate(frequencies):
        phase_increment = 2 * np.pi * freq * t_buff
        for n in range(2):
            audio[n] += vars[n][i] * np.sin(phase_increment + phases[n][i])
            phases[n][i] = (phases[n][i] + 2*np.pi*freq*block_size/sample_rate) % (2*np.pi)

    # Normalize
    max_val = max(np.abs(aud).max() for aud in audio)
    if max_val > 0: audio = [aud/max_val for aud in audio]
    # Return combined stereo
    outdata[:] = np.column_stack(audio).astype(np.float32)[:frames]
#endregion
#endregion

#region - Initalizating
#region -- Connecting to cameras
log(0, 'Connecting to cameras')
cameras = [cv2.VideoCapture(n) for n in camera_id]
if err:=sum((not cam.isOpened())<<n for n, cam in enumerate(cameras)):
    log(ERROR|EXIT, f"Could not open {['','left','right','both'][err]} camera/s")
# Eliminate buffer so that capture is of latest frame
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
#region -- Start audio
if False:
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

    # Create Window
    plt.ion()
    fig = plt.figure(figsize=(16,9), num='Thesis')
    # Define grid with top row taking 75% height
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
    # Second Row - Audio/Calibration
    brow = [fig.add_subplot(gs[1, n]) for n in range(2)]
    if len(calibration_out):
        brow[0].set_title(f'Calibration Error ({baseline_cm} cm)')
        brow[0].set_xlabel('Baseline Distance (cm)', fontsize=10)
        brow[0].set_ylabel('Reprojection Error (px)', fontsize=10)

        brow[1].set_title('Depth Map')
        brow[1].set_xlabel('Time (seconds)', fontsize=10)
        brow[1].set_ylabel('Frequency (Hz)', fontsize=10)
    else:
        brow[0].set_title('Left Audio Channel')
        brow[0].set_xlabel('Time (seconds)', fontsize=10)
        brow[0].set_ylabel('Frequency (Hz)', fontsize=10)
        brow[1].set_title('Right Audio Channel')
        brow[1].set_xlabel('Time (seconds)', fontsize=10)
        brow[1].set_ylabel('Frequency (Hz)', fontsize=10)
    # Combine All
    axes = np.array([camera_imgs, brow])
    fig.suptitle('Thesis Calibration' if len(calibration_out) else 'Thesis Demo', fontsize=16)
    # Start capture threads
    threads = [threading.Thread(target=capture_thread, args=(i, cam), daemon=True) for i, cam in enumerate(cameras)]
    for t in threads: t.start()
    # Animation function (called every interval ms)
    log(0, 'Starting main loop')
    #endregion
    #region --- Main loop thread
    def update_animation(frame_num):
        #region ---- Capturing camera
        # Read images
        cam_frames = [None] * 2
        for i in range(2):
            try:
                cam_frames[i] = frame_queues[i].get_nowait()
            except queue.Empty:
                pass  # Keep old frame if no new one
        # Check if we got frames - but don't return early, just log
        err = sum((cam_frames[n] is None)<<n for n in range(2))
        if err:
            log(WARNING,f"Failed to capture image from {['','left','right','both'][err]} camera/s")
        #endregion
        #region ---- Hand gesture recognition
        # Hand gesture detection in right camera
        if hand_gesture and not err:
            frame_rgb = cv2.cvtColor(cam_frames[1], cv2.COLOR_BGR2RGB)
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
        # Calibration mode (process on BGR before RGB conversion)
        if len(calibration_out):
            # Initialize defaults
            rets = [False, False]
            corners = [None, None]
            
            # Process each camera independently
            for n in range(2):
                if cam_frames[n] is not None:
                    gray = cv2.cvtColor(cam_frames[n], cv2.COLOR_BGR2GRAY)
                    ret, corner = cv2.findChessboardCorners(gray, chessboard, None)
                    rets[n] = ret
                    corners[n] = corner
                    
                    if ret:
                        cv2.drawChessboardCorners(cam_frames[n], chessboard, corner, ret)

            status = "READY" if all(rets) else "NO PATTERN"
            color = (0, 255, 0) if all(rets) else (0, 0, 255)

            if cam_frames[0] is not None:
                for n in range(2):
                    cv2.putText(cam_frames[0], f"CALIBRATION: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color if n else (0,0,0), 3-n)
                    cv2.putText(cam_frames[0], f"Captured: {len(calib_imgs[0])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*n,)*3, 3-n)
                    cv2.putText(cam_frames[0], "SPACE=Capture, ESC=Process", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*n,)*3, 3-n)
        #endregion
        #region ---- Output
        # Update displays (convert to RGB and set data)
        for i in range(2):
            if cam_frames[i] is not None:
                rgb_frame = cv2.cvtColor(cam_frames[i], cv2.COLOR_BGR2RGB)
                camera_imgs[i].set_data(rgb_frame)
        return camera_imgs  # Always return the Artist objects
        #endregion
    #endregion
    #region --- Ending
    # Create animation (blit=True for speed: only redraws changed parts)
    ani = FuncAnimation(fig, update_animation, interval=10, blit=True, cache_frame_data=False)  # 10ms ~100 FPS target, but actual ~30-60
    # Show and block until window closed (handles quit internally)
    plt.show(block=True)
    # Cleanup
    stop_event.set()
    for t in threads: t.join()
    for q in frame_queues: q.queue.clear()
    #endregion
#endregion
#endregion
