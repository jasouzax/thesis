import cv2
import numpy as np

class StereoDepthMap:
    def __init__(self, baseline=0.06, fov=160, resolution=(2592, 1944)):
        """
        Initialize stereo depth mapping system
        
        Args:
            baseline: Distance between cameras in meters (0.06m = 6cm)
            fov: Field of view in degrees (160°)
            resolution: Camera resolution (width, height) for 5MP
        """
        self.baseline = baseline
        self.fov = fov
        self.width, self.height = resolution
        
        # Calculate focal length from FOV
        self.focal_length = (self.width / 2) / np.tan(np.radians(fov / 2))
        
        # Initialize stereo matcher (SGBM - Semi-Global Block Matching)
        # More aggressive parameters for uncalibrated setup
        min_disp = -16  # Allow negative disparity for uncalibrated cameras
        num_disp = 16 * 12  # Must be divisible by 16, increased range
        block_size = 5  # Smaller for more detail
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=12,
            uniquenessRatio=5,  # Lower = less strict
            speckleWindowSize=50,
            speckleRange=16,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Simpler stereo matcher as backup
        self.stereo_simple = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
        
    def compute_depth(self, left_frame, right_frame, use_simple=False):
        """
        Compute depth map from stereo pair
        
        Args:
            left_frame: Left camera image
            right_frame: Right camera image
            use_simple: Use simpler BM algorithm instead of SGBM
            
        Returns:
            depth_map: Depth values in meters
            disparity_normalized: Normalized disparity for visualization
        """
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better matching
        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        
        # Compute disparity
        if use_simple:
            disparity = self.stereo_simple.compute(gray_left, gray_right)
        else:
            disparity = self.stereo.compute(gray_left, gray_right)
        
        disparity = disparity.astype(np.float32) / 16.0
        
        # Debug: Print disparity stats
        valid_disp = disparity[disparity > 0]
        if len(valid_disp) > 0:
            print(f"Disparity range: {valid_disp.min():.1f} to {valid_disp.max():.1f}, valid pixels: {len(valid_disp)}")
        else:
            print("WARNING: No valid disparity found!")
        
        # Convert disparity to depth: depth = (baseline * focal_length) / disparity
        depth_map = np.zeros_like(disparity)
        valid_mask = disparity > 0
        depth_map[valid_mask] = (self.baseline * self.focal_length) / disparity[valid_mask]
        
        # Normalize disparity for visualization (0-255)
        disparity_normalized = cv2.normalize(
            disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        
        return depth_map, disparity_normalized

def main():
    # Initialize cameras (adjust camera indices as needed)
    cap_left = cv2.VideoCapture(0)  # Left camera
    cap_right = cv2.VideoCapture(2)  # Right camera
    
    # Set camera resolution to half 5MP
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 2592//4)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944//4)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 2592//4)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944//4)
    
    # Check if cameras opened successfully
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open cameras")
        return
    
    # Initialize depth map generator
    stereo_depth = StereoDepthMap(baseline=0.06, fov=160)
    
    print("Press 'q' to quit, 's' to save depth map, 'b' to toggle BM/SGBM")
    print("Make sure both cameras can see the same scene!")
    
    use_simple = False
    
    while True:
        # Capture frames
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            print("Error: Failed to capture frames")
            break
        
        # Don't resize - already using half resolution
        # Compute depth map
        depth_map, disparity_vis = stereo_depth.compute_depth(
            frame_left, frame_right, use_simple
        )
        
        # Apply colormap for better visualization
        depth_colormap = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
        
        # Add text overlay showing algorithm
        algo_text = "Algorithm: BM (Simple)" if use_simple else "Algorithm: SGBM (Advanced)"
        cv2.putText(depth_colormap, algo_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stack images for display
        top_row = np.hstack((frame_left, frame_right))
        bottom_row = np.hstack((depth_colormap, depth_colormap))
        combined = np.vstack((top_row, bottom_row))
        
        # Display
        cv2.imshow('Stereo Depth Map (Top: L/R Cameras, Bottom: Depth)', combined)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('depth_map.png', depth_colormap)
            np.save('depth_values.npy', depth_map)
            print("Saved depth map")
        elif key == ord('b'):
            use_simple = not use_simple
            print(f"Switched to {'BM' if use_simple else 'SGBM'}")
    
    # Cleanup
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()