# IMPORTS
import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np
import time
import sys
import threading
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
import torch

# CONFIGURATIONS
def arg(deff, pref, run):
    return [*[run(arg[len(pref):]) for arg in sys.argv if arg.startswith(pref)],deff][0]

camera_id                   = arg(0, '-ci=', int)
sample_rate                 = arg(44100,'-sr=', int)
block_size                  = arg(512,  '-bs=', int)
num_bins                    = arg(64,   '-nb=', int)
history_duration            = arg(5.0, '-hd=', float)
echo_duration               = arg(0.0,  '-ed=', float)
max_depth_meters            = arg(5.0,  '-dm=', float)
min_volume                  = arg(0.01, '-mv=', float)
spectrogram_history         = arg(100, '-sh=', int)  # Number of time slices to keep
processing_scale            = arg(0.5, '-ps=', float)  # Scale factor for processing (0.5 = half size)
depth_skip_frames           = arg(2, '-ds=', int)  # Process depth every N frames
cycle_duration              = arg(4.0, '-cd=', float)  # Cycle duration in seconds

# SESSION DATA
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cycle_start_time = [time.time()]
current_slice_position = [0]  # Current x position in depth map (0 to width-1)
slice_direction = [1]  # 1 for left-to-right, -1 for right-to-left
frequencies = np.logspace(np.log10(100), np.log10(8000), num_bins)
time_block = np.arange(block_size) / sample_rate

# Load MiDaS depth model
print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print(f"MiDaS loaded on device: {device}")

# Shared data structures
spectrogram_lock = threading.Lock()
left_spectrogram = deque(maxlen=spectrogram_history)
right_spectrogram = deque(maxlen=spectrogram_history)
current_frame = [None]
current_depth = [None]
current_depth_gray = [None]  # Grayscale depth for audio generation
camera_shape = [None]
running = [True]  # Flag to control program execution
hand_gesture = [None]  # Current hand gesture: "open", "closed", "call", or None
last_call_time = [0]  # Last time "Calling..." was printed

# Initialize spectrograms with zeros
for _ in range(spectrogram_history):
    left_spectrogram.append(np.zeros(num_bins))
    right_spectrogram.append(np.zeros(num_bins))

# HAND GESTURE DETECTION
def is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    """Check if a finger is extended by comparing tip position to joints"""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    
    # Finger is extended if tip is further from MCP than PIP is
    tip_dist = np.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
    pip_dist = np.sqrt((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2)
    
    return tip_dist > pip_dist * 1.2

def detect_hand_gesture(hand_landmarks):
    """Detect specific hand gestures: open, closed, call"""
    landmarks = hand_landmarks.landmark
    
    # Check each finger extension status
    # Thumb: tip(4), IP(3), MCP(2)
    thumb_extended = landmarks[4].x < landmarks[3].x if landmarks[4].x < 0.5 else landmarks[4].x > landmarks[3].x
    
    # Index: tip(8), PIP(6), MCP(5)
    index_extended = is_finger_extended(landmarks, 8, 6, 5)
    
    # Middle: tip(12), PIP(10), MCP(9)
    middle_extended = is_finger_extended(landmarks, 12, 10, 9)
    
    # Ring: tip(16), PIP(14), MCP(13)
    ring_extended = is_finger_extended(landmarks, 16, 14, 13)
    
    # Pinky: tip(20), PIP(18), MCP(17)
    pinky_extended = is_finger_extended(landmarks, 20, 18, 17)
    
    # Count extended fingers
    extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Open hand: all 5 fingers extended
    if extended_count >= 4:  # Allow some tolerance
        return "open"
    
    # Closed hand (fist): no fingers extended
    if extended_count <= 1:
        return "closed"
    
    # Call gesture: thumb and pinky extended, others closed
    if thumb_extended and pinky_extended and not index_extended and not middle_extended and not ring_extended:
        return "call"
    
    # Any other configuration
    return None
def extract_depth_slice(depth_gray, x_pos):
    """Extract a vertical slice from depth map and convert to frequency coefficients"""
    if depth_gray is None:
        return np.zeros(num_bins)
    
    height, width = depth_gray.shape
    
    # Ensure x_pos is within bounds
    x_pos = max(0, min(x_pos, width - 1))
    
    # Extract vertical slice
    slice_data = depth_gray[:, x_pos]
    
    # Flip vertically so top = high frequency, bottom = low frequency
    slice_data = np.flip(slice_data)
    
    # Resize slice to match num_bins
    coefficients = cv2.resize(slice_data.reshape(-1, 1), (1, num_bins)).flatten()
    
    # Normalize to 0-1 range (depth value represents volume)
    # Invert so closer (brighter in depth map) = louder
    coefficients = coefficients.astype(np.float32) / 255.0
    
    # Scale to reasonable audio amplitude
    coefficients = coefficients * 0.1
    
    return coefficients

def generate_audio_from_spectrum(coefficients):
    """Generate audio signal from frequency coefficients"""
    audio = np.zeros(block_size)
    for i, (freq, amp) in enumerate(zip(frequencies, coefficients)):
        audio += amp * np.sin(2 * np.pi * freq * time_block)
    return audio

def audio_callback(outdata, frames, time_info, status):
    # Get current hand gesture
    current_gesture = hand_gesture[0]
    
    left_coeffs = np.zeros(num_bins)
    right_coeffs = np.zeros(num_bins)
    
    # Generate audio based on gesture
    if current_gesture == "open":
        # Open hand: cycle through depth map from right to left
        if current_depth_gray[0] is not None:
            current_time = time.time()
            elapsed = current_time - cycle_start_time[0]
            cycle_progress = (elapsed % cycle_duration) / cycle_duration  # 0 to 1
            
            width = current_depth_gray[0].shape[1]
            
            # Triangular wave: 0->1->0 over the cycle
            if cycle_progress < 0.5:
                # Left to right
                normalized_pos = cycle_progress * 2  # 0 to 1
                current_slice_position[0] = int(normalized_pos * (width - 1))
            else:
                # Right to left
                normalized_pos = (1 - cycle_progress) * 2  # 1 to 0
                current_slice_position[0] = int(normalized_pos * (width - 1))
            
            # Extract depth slice
            mono_coeffs = extract_depth_slice(current_depth_gray[0], current_slice_position[0])
            
            # Calculate stereo panning
            pan = current_slice_position[0] / (width - 1)  # 0 to 1
            left_gain = 1.0 - pan
            right_gain = pan
            
            left_coeffs = mono_coeffs * left_gain
            right_coeffs = mono_coeffs * right_gain
    
    elif current_gesture == "closed":
        # Closed hand: only output middle of depth map
        if current_depth_gray[0] is not None:
            width = current_depth_gray[0].shape[1]
            middle_pos = width // 2
            current_slice_position[0] = middle_pos
            
            # Extract middle slice
            mono_coeffs = extract_depth_slice(current_depth_gray[0], middle_pos)
            
            # Equal stereo (middle position)
            left_coeffs = mono_coeffs * 0.5
            right_coeffs = mono_coeffs * 0.5
    
    elif current_gesture == "call":
        # Call gesture: no audio output (silence)
        pass
    
    else:
        # No recognized gesture: no audio output (silence)
        pass
    
    # Store coefficients as spectrogram
    with spectrogram_lock:
        left_spectrogram.append(left_coeffs.copy())
        right_spectrogram.append(right_coeffs.copy())
    
    # Generate audio from coefficients
    audio_left = generate_audio_from_spectrum(left_coeffs)
    audio_right = generate_audio_from_spectrum(right_coeffs)
    
    outdata[:] = np.column_stack([audio_left, audio_right]).astype(np.float32)[:frames]

audio_stream = sd.OutputStream(
    samplerate=sample_rate,
    blocksize=block_size,
    channels=2,
    callback=audio_callback,
    dtype=np.float32
)

# DEPTH MAP GENERATION WITH MiDaS
def generate_depth_map(frame, process_size):
    """Generate depth map using MiDaS small model"""
    with torch.no_grad():
        # Resize for faster processing
        if processing_scale < 1.0:
            small_frame = cv2.resize(frame, process_size)
        else:
            small_frame = frame
            
        # Prepare input
        input_batch = transform(small_frame).to(device)
        
        # Predict depth
        prediction = midas(input_batch)
        
        # Resize back to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Convert to numpy
        depth = prediction.cpu().numpy()
        
        # Normalize to 0-255 for display and audio generation
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Create grayscale version for audio generation
        depth_gray = depth_normalized.copy()
        
        # Apply colormap for visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colored, depth_gray

# WINDOW SETUP (will be created after camera connects)
def setup_window():
    """Create and configure the matplotlib window"""
    plt.ion()
    fig = plt.figure(figsize=(10, 10), num='Hand Gesture Audio-Visual System')
    fig.patch.set_facecolor('white')
    grid = GridSpec(2, 2, hspace=0.1, wspace=0.1)
    
    # Create subplots
    ax_camera = fig.add_subplot(grid[0, 0])
    ax_depth = fig.add_subplot(grid[0, 1])
    ax_left_spec = fig.add_subplot(grid[1, 0])
    ax_right_spec = fig.add_subplot(grid[1, 1])
    
    # Configure axes
    for ax in [ax_camera, ax_depth]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    ax_camera.set_title('Camera + Hand Landmarks', fontsize=10)
    ax_depth.set_title('Depth Map (MiDaS)', fontsize=10)
    ax_left_spec.set_title('Left Channel Spectrogram', fontsize=10)
    ax_right_spec.set_title('Right Channel Spectrogram', fontsize=10)
    
    for ax in [ax_left_spec, ax_right_spec]:
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Frequency (Hz)', fontsize=8)
        ax.tick_params(labelsize=6)
    
    # Initialize with placeholder
    img_camera = ax_camera.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    img_depth = ax_depth.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    img_left_spec = ax_left_spec.imshow(
        np.zeros((num_bins, spectrogram_history)), 
        aspect='auto', 
        origin='lower',
        cmap='viridis',
        vmin=0, 
        vmax=0.1
    )
    img_right_spec = ax_right_spec.imshow(
        np.zeros((num_bins, spectrogram_history)), 
        aspect='auto', 
        origin='lower',
        cmap='viridis',
        vmin=0, 
        vmax=0.1
    )
    
    # Add vertical lines to show current slice position
    line_left = ax_left_spec.axvline(x=0, color='red', linewidth=2, alpha=0.7)
    line_right = ax_right_spec.axvline(x=0, color='red', linewidth=2, alpha=0.7)
    
    return fig, img_camera, img_depth, img_left_spec, img_right_spec, ax_left_spec, ax_right_spec, line_left, line_right

# CAMERA PROCESSING
def camera_thread():
    """Background thread for camera capture and processing"""
    cap = cv2.VideoCapture(camera_id)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        running[0] = False
        return
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_width}x{actual_height}")
    
    camera_shape[0] = (actual_height, actual_width)
    
    # Calculate processing size for depth map
    process_width = int(actual_width * processing_scale)
    process_height = int(actual_height * processing_scale)
    process_size = (process_width, process_height)
    
    frame_count = 0
    while running[0]:
        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: Failed to grab frame {frame_count}")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate depth map every N frames (skip frames for performance)
        if frame_count % depth_skip_frames == 0:
            try:
                depth_map, depth_gray = generate_depth_map(frame_rgb, process_size)
                current_depth[0] = depth_map
                current_depth_gray[0] = depth_gray
            except Exception as e:
                print(f"Depth generation error: {e}")
                if current_depth[0] is None:
                    current_depth[0] = np.zeros_like(frame)
                    current_depth_gray[0] = np.zeros((actual_height, actual_width), dtype=np.uint8)
        
        # Process hand landmarks
        results = hands.process(frame_rgb)
        
        # Detect hand gestures and openness
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Check if it's a right hand
                is_right_hand = handedness.classification[0].label == "Right"
                
                if is_right_hand:
                    # Detect gesture
                    gesture = detect_hand_gesture(hand_landmarks)
                    hand_gesture[0] = gesture
                    
                    # Print "Calling..." if call gesture detected (throttled to once per second)
                    if gesture == "call":
                        current_time = time.time()
                        if current_time - last_call_time[0] > 1.0:
                            print("Calling...")
                            last_call_time[0] = current_time
                    
                    # Draw landmarks on camera frame
                    mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                else:
                    # Not right hand, no audio
                    hand_gesture[0] = None
        else:
            # No hand detected
            hand_gesture[0] = None
        
        current_frame[0] = frame_rgb
    
    cap.release()
    print("Camera released")

# MAIN LOOP
try:
    # Start camera thread FIRST (before any GUI)
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()
    
    print("Waiting for camera to connect...")
    
    # Wait for first frame (camera to connect)
    timeout = 10  # seconds
    start_time = time.time()
    while current_frame[0] is None and running[0]:
        if time.time() - start_time > timeout:
            print("ERROR: Camera connection timeout")
            running[0] = False
            break
        time.sleep(0.1)
    
    if not running[0]:
        print("Exiting due to camera error")
        sys.exit(1)
    
    print("Camera connected! Starting audio and GUI...")
    
    # Start audio stream
    audio_stream.start()
    
    # Now create the window (after camera is ready)
    fig, img_camera, img_depth, img_left_spec, img_right_spec, ax_left_spec, ax_right_spec, line_left, line_right = setup_window()
    
    print("System running. Press 'q' to quit.")
    
    # FPS and gesture tracking
    fps_times = []
    fps_values = []
    gesture_events = []  # list of (time_from_start, gesture)
    loop_start_time = time.time()
    last_loop_time = loop_start_time
    last_gesture = None
    
    while running[0]:
        # Check for key press
        if plt.get_fignums():  # Check if window still exists
            # Check for 'q' key press
            if plt.waitforbuttonpress(timeout=0.001):
                print("Quit requested")
                running[0] = False
                break
        else:
            # Window was closed
            print("Window closed")
            running[0] = False
            break
        
        # Update camera and depth displays
        if current_frame[0] is not None:
            img_camera.set_data(current_frame[0])
        if current_depth[0] is not None:
            img_depth.set_data(cv2.cvtColor(current_depth[0], cv2.COLOR_BGR2RGB))
        
        # Update spectrograms
        with spectrogram_lock:
            left_data = np.array(left_spectrogram).T
            right_data = np.array(right_spectrogram).T
        
        img_left_spec.set_data(left_data)
        img_right_spec.set_data(right_data)
        
        # Update the vertical line indicating current slice position
        # The line should be at the rightmost position (most recent time)
        line_left.set_xdata([spectrogram_history - 1])
        line_right.set_xdata([spectrogram_history - 1])
        
        # Refresh display
        plt.pause(0.016)  # ~60 fps target
        
        # Track FPS
        current_loop_time = time.time()
        delta = current_loop_time - last_loop_time
        if delta > 0:
            fps = 1 / delta
        else:
            fps = 0
        time_from_start = current_loop_time - loop_start_time
        fps_times.append(time_from_start)
        fps_values.append(fps)
        last_loop_time = current_loop_time
        
        # Track gesture changes
        current_gesture = hand_gesture[0]
        if current_gesture != last_gesture:
            if current_gesture is not None:
                gesture_events.append((time_from_start, current_gesture))
            last_gesture = current_gesture
        
except KeyboardInterrupt:
    print("\nShutting down...")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    running[0] = False
    print("Stopping audio...")
    audio_stream.stop()
    audio_stream.close()
    print("Closing hand detector...")
    hands.close()
    print("Closing plots...")
    plt.close('all')
    # Generate FPS graph if data exists
    if fps_times:
        fig_fps = plt.figure(figsize=(10, 5))
        ax = fig_fps.add_subplot(111)
        ax.plot(fps_times, fps_values, label='FPS')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('FPS')
        ax.set_title('FPS over Time')
        # Add gesture markers
        colors = {'open': 'green', 'closed': 'red', 'call': 'blue'}
        for t, g in gesture_events:
            ax.axvline(x=t, color=colors.get(g, 'black'), linestyle='--', label=g)
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.savefig('fps.png')
        plt.close(fig_fps)
        print("Saved FPS graph to fps.png")
    print("Cleanup complete. Exiting.")
    sys.exit(0)