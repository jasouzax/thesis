#region Imports
#region Environment
import sys                                                                  # System - For CLI arguments
import os                                                                   # OS - For Filesystem processing
import time                                                                 # Time - For period/time
#endregion
#region Libraries
import cv2                                                                  # OpenCV2 - Image Processing
# import mediapipe as mp                                                      # MediaPipe - Hand Gesture Recognition
import numpy as np                                                          # Numpy - Data processing
# import torch                                                                # Torch - AI Model Execution
import sounddevice as sd                                                    # SoundDevice - For live audio generation
import matplotlib.pyplot as plt                                             # MatPlotLib - Rendering graphs
#endregion
#region Library Typing
from typing import cast, NamedTuple, Optional, List, Any, TypeVar, Callable # General Python types
from dataclasses import dataclass                                           # Data Classes
# from mediapipe.python.solutions import hands as mp_hands                    # MediaPipe Hands
# from mediapipe.python.solutions import drawing_utils as mp_drawing          # MediaPipe Rendering Hand Points
# from mediapipe.python.solutions import drawing_styles as mp_drawing_styles  # MediaPipe Rendering Hand Points
from numpy.typing import NDArray                                            # Numpy Array type
#endregion
#endregion

#region Helper functions
#region Argument parsing
T = TypeVar('T')
def arg(deff:T, pref:str, run:Callable[[str],T]=lambda x:x) -> T:
    return [*[run(arg[len(pref):]) for arg in sys.argv if arg.startswith(pref)],deff][0]
#endregion
#region Logging
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
#endregion

#region Configuration
#region Debugging
gui_window:bool             = arg(False,'-gw=', bool)
#endregion
#region Calibration
chessboard:tuple[int,int]   = arg([9,6],'-bd=', lambda t:[int(v) for v in t.split('x')])
square_size:int             = arg(19,   '-sq=', int)
camera_id:tuple[int,int]    = arg([0,2],'-ci=', lambda t:[int(v) for v in t.split(',')])
img_scale:float             = arg(0.5,  '-sc=', float)
calibration_out:str         = arg('', '-co=')
#endregion
#region Audio
sample_rate:int             = arg(44100,'-sr=', int)
block_size:int              = arg(512,  '-bs=', int)
num_bins:int                = arg(64,   '-nb=', int)
history_duration:float      = arg(10.0, '-hd=', int)
#endregion
#region Session data
calib_imgs:tuple[List[NDArray],List[NDArray]] = [[],[]]
time_counter:tuple[float]   = [0]
time_start:float            = time.time()
fig, (axaud_l, axaud_r)     = plt.subplots(2, 1, figsize=(12,8))
#endregion
#endregion

#region Main functions
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
        # Gererate audio
        audio = [audio[n] + vars[n][i] * np.sin(phase_increment + phases[n][i]) for n in range(2)]
        # Accumate phases within rage [0,2pi]
        for phase in phases: phase[i] = (phase[i]+2*np.pi*freq*block_size/sample_rate)%(2*np.pi)
    # Normalize
    max_val = max(np.abs(aud).max() for aud in audio)
    if max_val > 0: audio = [aud/max_val for aud in audio]
    # Return combined stereo
    outdata[:] = np.column_stack(audio).astype(np.float32)[:frames]

#endregion

#region Initalizating
#region Cameras
# Connect to cameras
log(0, 'Connecting to cameras')
cameras = [cv2.VideoCapture(n) for n in camera_id]
if err:=sum((not cam.isOpened())<<n for n, cam in enumerate(cameras)):
    log(ERROR|EXIT, f"Could not open {['','left','right','both'][err]} camera/s")
# Eliminate buffer so that capture is of latest frame
log(0, 'Eliminating camera buffer')
for camera in cameras: camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# Get camera dimensions
log(0, 'Getting camera dimensions')
init_frames = cast(tuple[tuple[bool,NDArray[np.uint8]],tuple[bool,NDArray[np.uint8]]], [camera.read() for camera in cameras])
if err:=sum((not res[0])<<n for n, res in enumerate(init_frames)):
    log(ERROR|EXIT, f"Could not read {['','left','right','both'][err]} camera/s")
if err:=sum((init_frames[0][1].shape[:2][n] != init_frames[1][1].shape[:2][n])<<n for n in range(2)):
    log(ERROR|EXIT, f"Camera {['','width', 'height', 'dimension']} does't match")
width, height = [int(dim*img_scale) for dim in init_frames[0][1].shape[:2]]
#endregion
#region Audio
# Frequency bns in logarithmic spacing for more perceptual relevance
frequencies = np.logspace(np.log10(100), np.log10(8000), num_bins)
# Time array for one block
time_block = np.arange(block_size) / sample_rate
# Audio stream
audio_stream = sd.OutputStream(
    samplerate=sample_rate,
    blocksize=block_size,
    channels=2,
    callback=audio_callback,
    dtype=np.float32
)
#audio_stream.start()
#endregion
#region Window
if gui_window:
    cv2.namedWindow('Thesis', cv2.WINDOW_NORMAL)
    #plt.show()
    while True:
        # Grab most recent frame
        for cam in cameras: cam.grab()
        # Read frame
        rets, frames = zip(*[cam.retrieve() for cam in cameras])
        if err:=sum((not ret)<<n for n, ret in enumerate(rets)):
            log(ERROR|EXIT, f"Failed to capture image")
        # Calibration mode
        if len(calibration_out):
            # Make image grayscale
            grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
            # Recognize chessboard
            rets, corners = zip(*[cv2.findChessboardCorners(gray, chessboard, None) for gray in grays])
            for n in range(2):
                if rets[n]: cv2.drawChessboardCorners(frames[n], chessboard, corners[n], rets[n])
            # Status of recognition
            status = "READY" if all(rets) else "NO PATTERN"
            color = (0,255,0) if all(rets) else (0,0,255)
            # Display status
            cv2.putText(frames[0], f"CALIBRATION: {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frames[0], f"Captured: {len(calib_imgs[0])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frames[0], "SPACE=Capture, ESC=Process", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Display results
        cv2.imshow('Thesis', np.hstack(frames))
        # Get key pressed
        key = cv2.waitKey(1) & 0xFF
        # User quit
        if key == ord('q') or key == 27:
            break

#endregion
#endregion
